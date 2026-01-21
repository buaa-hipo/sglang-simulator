import os
from typing import Dict, List, Optional
import setproctitle
import faulthandler
import signal
import psutil
import threading
import torch
import zmq
from typing import Any
from sglang.srt.utils import get_zmq_socket
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.server_args import PortArgs, ServerArgs, get_global_server_args
from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_recv_skipper import SchedulerRecvSkipper
from sglang.srt.managers.scheduler_input_blocker import SchedulerInputBlocker
from sglang.srt.utils import (
    configure_gc_logger,
    configure_logger,
    get_available_gpu_memory,
    get_bool_env_var,
    kill_itself_when_parent_died,
    numa_bind_to_node,
    require_mlp_sync,
    set_gpu_proc_affinity,
    set_random_seed,
    suppress_other_loggers,
)
from sglang.srt.tracing.trace import (
    process_tracing_init,
    trace_set_thread_info,
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
)
from sglang.utils import get_exception_traceback
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.distributed import get_pp_group, get_world_group
from torch.cuda import Stream as CudaStream
from sglang.srt.managers.session_controller import Session
from sglang.utils import TypeBasedDispatcher, get_exception_traceback
from sglang.srt.managers.schedule_batch import (
    Req,
    ScheduleBatch,
)
from sglang.srt.constrained.base_grammar_backend import (
    create_grammar_backend,
)
from sglang.srt.managers.schedule_policy import (
    SchedulePolicy,
)
from sglang.srt.managers.io_struct import (
    AbortReq,
    BaseBatchReq,
    BaseReq,
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    CheckWeightsReqInput,
    ClearHiCacheReqInput,
    ClearHiCacheReqOutput,
    CloseSessionReqInput,
    ContinueGenerationReqInput,
    DestroyWeightsUpdateGroupReqInput,
    ExpertDistributionReq,
    ExpertDistributionReqOutput,
    ExpertDistributionReqType,
    FlushCacheReqInput,
    FlushCacheReqOutput,
    FreezeGCReq,
    GetInternalStateReq,
    GetInternalStateReqOutput,
    GetLoadReqInput,
    GetWeightsByNameReqInput,
    HealthCheckOutput,
    InitWeightsSendGroupForRemoteInstanceReqInput,
    InitWeightsSendGroupForRemoteInstanceReqOutput,
    InitWeightsUpdateGroupReqInput,
    LoadLoRAAdapterReqInput,
    LoadLoRAAdapterReqOutput,
    OpenSessionReqInput,
    OpenSessionReqOutput,
    PauseGenerationReqInput,
    ProfileReq,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    RpcReqInput,
    RpcReqOutput,
    SendWeightsToRemoteInstanceReqInput,
    SendWeightsToRemoteInstanceReqOutput,
    SetInternalStateReq,
    SetInternalStateReqOutput,
    SlowDownReqInput,
    SlowDownReqOutput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    UnloadLoRAAdapterReqInput,
    UnloadLoRAAdapterReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.scheduler import (
    Scheduler,
    logger,
)

from sglang.simulator.managers.controller import get_simulation_controller


class SchedulerSimulation(Scheduler):  # 劫持父类，重写其方法
    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        tp_rank: int,
        moe_ep_rank: int,
        pp_rank: int,
        dp_rank: Optional[int],
        profiler_push_ipc_name: str
    ):

        self.profiler_push_ipc_name = profiler_push_ipc_name

        # Parse args
        self.server_args = server_args
        self.tp_rank = tp_rank
        self.moe_ep_rank = moe_ep_rank
        self.pp_rank = pp_rank
        self.dp_rank = dp_rank
        self.tp_size = server_args.tp_size
        self.moe_ep_size = server_args.ep_size
        self.pp_size = server_args.pp_size
        self.dp_size = server_args.dp_size
        self.schedule_policy = server_args.schedule_policy
        self.enable_priority_scheduling = server_args.enable_priority_scheduling
        self.abort_on_priority_when_disabled = (
            server_args.abort_on_priority_when_disabled
        )
        self.schedule_low_priority_values_first = (
            server_args.schedule_low_priority_values_first
        )
        self.priority_scheduling_preemption_threshold = (
            server_args.priority_scheduling_preemption_threshold
        )
        self.enable_lora = server_args.enable_lora
        self.max_loras_per_batch = server_args.max_loras_per_batch
        self.enable_overlap = not server_args.disable_overlap_schedule
        self.enable_pdmux = server_args.enable_pdmux
        self.skip_tokenizer_init = server_args.skip_tokenizer_init
        self.enable_metrics = server_args.enable_metrics
        self.enable_metrics_for_all_schedulers = (
            server_args.enable_metrics_for_all_schedulers
        )
        self.enable_kv_cache_events = bool(
            server_args.kv_events_config and tp_rank == 0
        )
        self.enable_trace = server_args.enable_trace
        self.stream_interval = server_args.stream_interval
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.gpu_id = gpu_id
        self.page_size = server_args.page_size
        self.enable_hierarchical_cache = server_args.enable_hierarchical_cache
        self.enable_hicache_storage = server_args.hicache_storage_backend is not None
        self.max_recv_per_poll = envs.SGLANG_SCHEDULER_MAX_RECV_PER_POLL.get()

        # Distributed rank info
        self.attn_tp_rank, self.attn_tp_size, self.attn_dp_rank = (
            compute_dp_attention_world_info(
                server_args.enable_dp_attention,
                self.tp_rank,
                self.tp_size,
                self.dp_size,
            )
        )

        # Init model config
        self.model_config = ModelConfig.from_server_args(server_args)

        # Init diffusion LLM config
        self.dllm_config = DllmConfig.from_server_args(server_args)

        # Init inter-process communication
        self.init_sockets(server_args, port_args)

        # Init pdmux context
        if self.enable_pdmux:
            self.init_pdmux()

        # Init tokenizer
        self.init_tokenizer()

        # Init moe config
        self.init_moe_config()

        # Init GEMM config (FP8 GEMM, etc.)
        self.init_gemm_config()

        # Check whether overlap can be enabled
        if not self.is_generation:
            self.enable_overlap = False
            logger.info("Overlap scheduler is disabled for embedding models.")
        
        context = zmq.Context.instance()
        self.perf_sender = get_zmq_socket(
            context, zmq.PUSH, profiler_push_ipc_name, False,  
        )  # 连接到Profiler进程

        # Launch a tensor parallel worker
        # *Simulation
        from sglang.simulator.srt.managers.tp_worker import TpModelWorkerSimulator

        self.tp_worker = TpModelWorkerSimulator(
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            moe_ep_rank=moe_ep_rank,
            pp_rank=pp_rank,
            dp_rank=dp_rank,
            nccl_port=port_args.nccl_port,
        )

        # Launch a draft worker for speculative decoding
        draft_worker_kwargs = dict(
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            moe_ep_rank=moe_ep_rank,
            server_args=server_args,
            nccl_port=port_args.nccl_port,
            target_worker=self.tp_worker,
            dp_rank=dp_rank,
        )

        if server_args.speculative_draft_load_format is not None:
            server_args.load_format = server_args.speculative_draft_load_format
            logger.info(
                f"Using draft model load_format: '{server_args.speculative_draft_load_format}'"
            )

        # Draft workers are looked up via `SpeculativeAlgorithm` registry; new
        # algorithms should register their factory instead of patching this code.
        if self.spec_algorithm.is_eagle():
            draft_worker_kwargs["enable_overlap"] = self.enable_overlap
        self.draft_worker = self.spec_algorithm.create_draft_worker(
            **draft_worker_kwargs
        )

        # Dispatch the model worker
        if self.spec_algorithm.is_none():
            self.model_worker = self.tp_worker
        else:
            self.model_worker = self.draft_worker

        # Get token and memory info from the model worker
        (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_queued_requests,
            self.max_req_len,
            self.max_req_input_len,
            self.random_seed,
            self.device,
            _,
            _,
            _,
        ) = self.tp_worker.get_worker_info()
        if get_global_server_args().pp_max_micro_batch_size is None:
            get_global_server_args().pp_max_micro_batch_size = max(
                self.max_running_requests // server_args.pp_size, 1
            )

        self.tp_group = self.tp_worker.get_tp_group()
        self.tp_cpu_group = self.tp_group.cpu_group
        self.attn_tp_group = self.tp_worker.get_attention_tp_group()
        self.attn_tp_cpu_group = self.tp_worker.get_attention_tp_cpu_group()
        self.pp_group = get_pp_group()
        self.world_group = get_world_group()

        # With DP attention enabled, the entry rank is attn_tp_rank==0;
        # otherwise the entry rank is TP group local rank 0.
        # For #11910, use the CPU communication group to broadcast VLM Python objects,
        # avoiding any coupling with CUDA streams/devices.
        if self.server_args.enable_dp_attention:
            self.cpu_group = self.attn_tp_cpu_group
            self.entry_rank = self.attn_tp_group.first_rank
            self.is_entry_rank = self.attn_tp_rank == 0
        else:
            self.cpu_group = self.tp_cpu_group
            self.entry_rank = self.tp_group.first_rank
            self.is_entry_rank = self.tp_group.rank_in_group == 0

        self.pad_input_ids_func = self.tp_worker.get_pad_input_ids_func()
        set_random_seed(self.random_seed)

        # Hybrid memory pool
        self.is_hybrid_swa = self.tp_worker.is_hybrid_swa
        self.is_ssm_model = (
            self.tp_worker.model_runner.hybrid_gdn_config is not None
            or self.tp_worker.model_runner.mamba2_config is not None
        )

        if self.is_hybrid_swa:
            self.sliding_window_size = self.tp_worker.sliding_window_size
            self.full_tokens_per_layer, self.swa_tokens_per_layer = (
                self.tp_worker.get_tokens_per_layer_info()
            )

        # Print debug info
        if tp_rank == 0:
            avail_mem = get_available_gpu_memory(
                self.device, self.gpu_id, empty_cache=False
            )
            logger.info(
                f"max_total_num_tokens={self.max_total_num_tokens}, "
                f"chunked_prefill_size={server_args.chunked_prefill_size}, "
                f"max_prefill_tokens={self.max_prefill_tokens}, "
                f"max_running_requests={self.max_running_requests}, "
                f"context_len={self.model_config.context_len}, "
                f"{'available_cpu_mem' if self.device == 'cpu' else 'available_gpu_mem'}={avail_mem:.2f} GB"
            )

        # Init metrics stats
        self.init_metrics(tp_rank, pp_rank, dp_rank)

        # Init cache using the existing memory pool
        self.init_cache_with_memory_pool()

        # Init running status
        self.waiting_queue: List[Req] = []
        # The running decoding batch for continuous batching
        self.running_batch: ScheduleBatch = ScheduleBatch(reqs=[], batch_is_full=False)
        # The current forward batch
        self.cur_batch: Optional[ScheduleBatch] = None
        # The current split prefill batch
        self.split_prefill_batch: Optional[ScheduleBatch] = None
        # The last forward batch
        self.last_batch: Optional[ScheduleBatch] = None
        self.forward_ct = 0
        self.forward_ct_decode = 0
        self.num_generated_tokens = 0
        self.last_prefill_tokens = 0
        self.return_health_check_ct = 0
        self.num_retracted_reqs: int = 0
        self.num_paused_reqs: int = 0
        self.sessions: Dict[str, Session] = {}
        self.default_stream: CudaStream = torch.get_device_module(
            self.device
        ).current_stream()
        if self.device == "cpu":
            self.default_stream.synchronize = lambda: None  # No-op for CPU
        self.forward_sleep_time = None
        self._engine_paused = False

        # Init chunked prefill
        self.chunked_prefill_size = server_args.chunked_prefill_size
        if self.dllm_config is not None:
            # We currently leverage chunked prefill to implement block diffusion
            # for diffusion LLM.
            self.chunked_prefill_size = self.dllm_config.block_size
        if self.chunked_prefill_size <= 0:  # -1 means disable
            self.chunked_prefill_size = None
        self.chunked_req = None
        self.is_mixed_chunk = (
            self.chunked_prefill_size is not None and server_args.enable_mixed_chunk
        )

        # Init the grammar backend for constrained generation
        self.grammar_queue: List[Req] = []
        if not server_args.skip_tokenizer_init:
            self.grammar_backend = create_grammar_backend(
                server_args,
                self.tokenizer,
                self.model_config.vocab_size,
                self.model_config.hf_eos_token_id,
            )
        else:
            self.grammar_backend = None

        # Init schedule policy and new token estimation
        self.policy = SchedulePolicy(
            self.schedule_policy,
            self.tree_cache,
            self.enable_hierarchical_cache,
            self.enable_priority_scheduling,
            self.schedule_low_priority_values_first,
        )
        # Enable preemption for priority scheduling.
        self.try_preemption = self.enable_priority_scheduling
        self.init_new_token_ratio = min(
            envs.SGLANG_INIT_NEW_TOKEN_RATIO.get()
            * server_args.schedule_conservativeness,
            1.0,
        )
        self.min_new_token_ratio = min(
            self.init_new_token_ratio * envs.SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR.get(),
            1.0,
        )
        self.new_token_ratio_decay = (
            self.init_new_token_ratio - self.min_new_token_ratio
        ) / envs.SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS.get()
        self.new_token_ratio = self.init_new_token_ratio

        # Init watchdog thread
        self.watchdog_timeout = server_args.watchdog_timeout
        t = threading.Thread(target=self.watchdog_thread, daemon=True)
        t.start()
        self.parent_process = psutil.Process().parent()

        # Init memory saver, profiler and metric stats
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=server_args.enable_memory_saver
        )
        self.offload_tags = set()
        self.init_profiler()
        self.recv_skipper = SchedulerRecvSkipper.maybe_create(server_args)
        self.input_blocker = (
            SchedulerInputBlocker(noop=self.attn_tp_rank != 0)
            if get_bool_env_var("SGLANG_ENABLE_COLOCATED_BATCH_GEN")
            else None
        )

        # Init disaggregation
        self.init_disaggregation()

        if self.enable_kv_cache_events:
            self.init_kv_events(server_args.kv_events_config)

        if envs.SGLANG_LOG_GC.get():
            configure_gc_logger()

        # Init prefill kv split size when deterministic inference is enabled with various attention backends
        self.init_deterministic_inference_config()

        # Init overlap
        self.init_overlap()

        # Init mlp sync flag
        self.require_mlp_sync = require_mlp_sync(server_args)

        # Init request dispatcher
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (TokenizedGenerateReqInput, self.handle_generate_request),
                (TokenizedEmbeddingReqInput, self.handle_embedding_request),
                (BatchTokenizedGenerateReqInput, self.handle_batch_generate_request),
                (BatchTokenizedEmbeddingReqInput, self.handle_batch_embedding_request),
                (FlushCacheReqInput, self.flush_cache_wrapped),
                (ClearHiCacheReqInput, self.clear_hicache_storage_wrapped),
                (AbortReq, self.abort_request),
                (OpenSessionReqInput, self.open_session),
                (CloseSessionReqInput, self.close_session),
                (UpdateWeightFromDiskReqInput, self.update_weights_from_disk),
                (InitWeightsUpdateGroupReqInput, self.init_weights_update_group),
                (DestroyWeightsUpdateGroupReqInput, self.destroy_weights_update_group),
                (
                    InitWeightsSendGroupForRemoteInstanceReqInput,
                    self.init_weights_send_group_for_remote_instance,
                ),
                (
                    SendWeightsToRemoteInstanceReqInput,
                    self.send_weights_to_remote_instance,
                ),
                (
                    UpdateWeightsFromDistributedReqInput,
                    self.update_weights_from_distributed,
                ),
                (UpdateWeightsFromTensorReqInput, self.update_weights_from_tensor),
                (UpdateWeightsFromIPCReqInput, self.update_weights_from_ipc),
                (GetWeightsByNameReqInput, self.get_weights_by_name),
                (ReleaseMemoryOccupationReqInput, self.release_memory_occupation),
                (ResumeMemoryOccupationReqInput, self.resume_memory_occupation),
                (CheckWeightsReqInput, self.check_weights),
                (SlowDownReqInput, self.slow_down),
                (ProfileReq, self.profile),
                (FreezeGCReq, self.handle_freeze_gc),
                (GetInternalStateReq, self.get_internal_state),
                (SetInternalStateReq, self.set_internal_state),
                (RpcReqInput, self.handle_rpc_request),
                (ExpertDistributionReq, self.expert_distribution_handle),
                (LoadLoRAAdapterReqInput, self.load_lora_adapter),
                (UnloadLoRAAdapterReqInput, self.unload_lora_adapter),
                (GetLoadReqInput, self.get_load),
                (PauseGenerationReqInput, self.pause_generation),
                (ContinueGenerationReqInput, self.continue_generation),
            ]
        )

    def scheduler_send_perf(self, perf: Any):
        print("scehduler sending perf")
        try:
            self.perf_sender.send_pyobj(perf, zmq.NOBLOCK)
        except zmq.Again:
            print("Drop perf")

    def event_loop_normal(self):
        """A normal scheduler loop."""

        # simulator_controller = get_simulation_controller()
        print("Entering event loop.")

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)

            recv_t = time.perf_counter()
            print("Scheduler received requests.")

            if self._engine_paused:
                continue

            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            batch_t = time.perf_counter()
            duration_recv = batch_t - recv_t
            self.scheduler_send_perf({
                "event": "host_scheduler_latency",
                "latency": duration_recv
            })  # 调度耗时

            print(f'Scheduler processing batch of size {0 if batch is None else len(batch.reqs)}')

            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)

                print('Scheduler finished processing batch.')

                result_t = time.perf_counter()
                self.scheduler_send_perf({
                    "event": "Operator_distribution_processing",
                    "latency": result_t - batch_t
                })   # 算子下发/处理

            else:
                # When the server is idle, do self-check and re-init some states
                self.self_check_during_idle()

            self.last_batch = batch

            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                self.self_check_during_busy()


def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    moe_ep_rank: int,
    pp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
    profiler_push_ipc_name: str, 
):
    # Generate the logger prefix
    prefix = ""
    if dp_rank is None and "SGLANG_DP_RANK" in os.environ:
        # [For Router] if env var "SGLANG_DP_RANK" exist, set dp_rank to the value of the env var
        dp_rank = int(os.environ["SGLANG_DP_RANK"])
    if dp_rank is not None:
        prefix += f" DP{dp_rank}"
    if server_args.pp_size > 1:
        prefix += f" PP{pp_rank}"
    if server_args.tp_size > 1:
        prefix += f" TP{tp_rank}"
    if server_args.ep_size > 1:
        prefix += f" EP{moe_ep_rank}"

    # Config the process
    setproctitle.setproctitle(f"sglang::scheduler{prefix.replace(' ', '_')}")
    faulthandler.enable()
    kill_itself_when_parent_died()
    parent_process = psutil.Process().parent()

    # Configure the logger
    configure_logger(server_args, prefix=prefix)
    suppress_other_loggers()

    # Set cpu affinity to this gpu process
    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(
            server_args.pp_size, server_args.tp_size, server_args.nnodes, gpu_id
        )
    if (
        numa_node := server_args.numa_node
    ) is not None and not envs.SGLANG_NUMA_BIND_V2.get():
        numa_bind_to_node(numa_node[gpu_id])

    # Set up tracing
    if server_args.enable_trace:
        process_tracing_init(server_args.otlp_traces_endpoint, "sglang")
        thread_label = "Scheduler"
        if server_args.disaggregation_mode == "prefill":
            thread_label = "Prefill Scheduler"
        elif server_args.disaggregation_mode == "decode":
            thread_label = "Decode Scheduler"
        trace_set_thread_info(thread_label, tp_rank, dp_rank)

    # Create a scheduler and run the event loop
    try:
        scheduler = SchedulerSimulation(
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            moe_ep_rank,
            pp_rank,
            dp_rank,
            profiler_push_ipc_name = profiler_push_ipc_name,
        )
        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": scheduler.max_total_num_tokens,
                "max_req_input_len": scheduler.max_req_input_len,
            }
        )

        disaggregation_mode: DisaggregationMode = scheduler.disaggregation_mode
        if disaggregation_mode == DisaggregationMode.NULL:
            if scheduler.enable_pdmux:
                scheduler.event_loop_pdmux()
                print("Scheduler entered pdmux event loop.")
            elif server_args.pp_size > 1:
                scheduler.event_loop_pp()
                print("Scheduler entered pp event loop.")
            elif scheduler.enable_overlap:
                scheduler.event_loop_overlap()
                print("Scheduler entered overlap event loop.")
            else:
                scheduler.event_loop_normal()
                print("Scheduler entered normal event loop.")
        elif disaggregation_mode == DisaggregationMode.PREFILL:
            if scheduler.enable_overlap:
                # TODO: Prefill节点
                scheduler.event_loop_overlap_disagg_prefill()
                print("Scheduler entered overlap disagg prefill event loop.")
            else:
                if server_args.pp_size > 1:
                    scheduler.event_loop_pp_disagg_prefill()
                    print("Scheduler entered pp disagg prefill event loop.")
                else:
                    scheduler.event_loop_normal_disagg_prefill()
                    print("Scheduler entered normal disagg prefill event loop.")

        elif disaggregation_mode == DisaggregationMode.DECODE:
            if scheduler.enable_overlap:
                # TODO: Decode节点
                scheduler.event_loop_overlap_disagg_decode()
                print("Scheduler entered overlap disagg decode event loop.")
            else:
                scheduler.event_loop_normal_disagg_decode()
                print("Scheduler entered normal disagg decode event loop.")

    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
