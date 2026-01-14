import os
import multiprocessing as mp
from typing import Dict, Optional, Tuple
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.simulator.srt.managers.scheduler import run_scheduler_process
from sglang.srt.managers.template_manager import TemplateManager
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.managers.data_parallel_controller import (
    run_data_parallel_controller_process,
)
from sglang.srt.managers.detokenizer_manager import run_detokenizer_process
from sglang.srt.managers.multi_tokenizer_mixin import MultiTokenizerRouter
from sglang.srt.utils import (
    configure_logger,
    launch_dummy_health_check_server,
    maybe_reindex_device_id,
    numa_utils,
    prepare_model_and_tokenizer,
)
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.entrypoints.engine import (
    _set_envs_and_config,
    logger,
    _init_tokenizer_manager,
)
from sglang.simulator.managers.controller import get_simulation_controller
from sglang.simulator.profiler.profiler import Profiler


def _launch_subprocesses(
    server_args: ServerArgs, port_args: Optional[PortArgs] = None
) -> Tuple[TokenizerManager, TemplateManager, Dict, PortArgs]:
    """
    Launch the TokenizerManager in the main process, the Scheduler in a subprocess, and the DetokenizerManager in another subprocess.
    """
    # *Simualtion Profiler
    start_time = Profiler.profile_time_host()

    # Configure global environment
    configure_logger(server_args)
    _set_envs_and_config(server_args)
    server_args.check_server_args()

    # Allocate ports for inter-process communications
    if port_args is None:
        port_args = PortArgs.init_new(server_args)
        logger.info(f"{server_args=}")

    # If using model from www.modelscope.cn, first download the model.
    server_args.model_path, server_args.tokenizer_path = prepare_model_and_tokenizer(
        server_args.model_path, server_args.tokenizer_path
    )

    # *Simulation
    simulator_controller = get_simulation_controller()

    scheduler_procs = []
    if server_args.dp_size == 1:
        # Launch tensor parallel scheduler processes
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=server_args.enable_memory_saver
        )
        scheduler_pipe_readers = []

        pp_size_per_node = max(server_args.pp_size // server_args.nnodes, 1)
        nnodes_per_pp_rank = max(server_args.nnodes // server_args.pp_size, 1)
        pp_rank_range = range(
            pp_size_per_node * (server_args.node_rank // nnodes_per_pp_rank),
            pp_size_per_node * (server_args.node_rank // nnodes_per_pp_rank + 1),
        )

        nnodes_per_tp_group = nnodes_per_pp_rank
        tp_size_per_node = server_args.tp_size // nnodes_per_tp_group
        tp_rank_range = range(
            tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group),
            tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group + 1),
        )

        for pp_rank in pp_rank_range:
            for tp_rank in tp_rank_range:
                reader, writer = mp.Pipe(duplex=False)
                gpu_id = (
                    server_args.base_gpu_id
                    + ((pp_rank % pp_size_per_node) * tp_size_per_node)
                    + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
                )
                moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)

                # *Simulation
                # TODO: TP和GPU映射
                tp_rank_mapped = simulator_controller.tp_mapping(tp_rank)
                gpu_id_mapped = simulator_controller.gpu_id_mapping(tp_rank_mapped)

                with maybe_reindex_device_id(gpu_id) as gpu_id:
                    proc = mp.Process(
                        target=run_scheduler_process,   # *Simulation
                        args=(
                            server_args,
                            port_args,
                            gpu_id,
                            tp_rank,
                            moe_ep_rank,
                            pp_rank,
                            None,
                            writer,
                        ),
                    )
                    with memory_saver_adapter.configure_subprocess(), numa_utils.configure_subprocess(
                        server_args, gpu_id
                    ):
                        proc.start()

                scheduler_procs.append(proc)
                scheduler_pipe_readers.append(reader)
    else:
        raise Exception("DP is not implemented!")
        # Launch the data parallel controller
        # reader, writer = mp.Pipe(duplex=False)
        # scheduler_pipe_readers = [reader]
        # proc = mp.Process(
        #     target=run_data_parallel_controller_process,
        #     args=(server_args, port_args, writer),
        # )
        # proc.start()
        # scheduler_procs.append(proc)

    if server_args.node_rank >= 1:
        # In multi-node cases, non-zero rank nodes do not need to run tokenizer or detokenizer,
        # so they can just wait here.

        for reader in scheduler_pipe_readers:
            data = reader.recv()
            assert data["status"] == "ready"

        if os.getenv("SGLANG_BLOCK_NONZERO_RANK_CHILDREN") == "0":
            # When using `Engine` as a Python API, we don't want to block here.
            return None, None, None, port_args

        launch_dummy_health_check_server(
            server_args.host, server_args.port, server_args.enable_metrics
        )

        for proc in scheduler_procs:
            proc.join()
            logger.error(
                f"Scheduler or DataParallelController {proc.pid} terminated with {proc.exitcode}"
            )
        return None, None, None, port_args

    # Launch detokenizer process
    detoken_proc = mp.Process(
        target=run_detokenizer_process,
        args=(
            server_args,
            port_args,
        ),
    )
    detoken_proc.start()

    # Init tokenizer manager first, as the bootstrap server is initialized here
    if server_args.tokenizer_worker_num == 1:
        tokenizer_manager, template_manager = _init_tokenizer_manager(
            server_args, port_args
        )
    else:
        # Launch multi-tokenizer router
        tokenizer_manager = MultiTokenizerRouter(server_args, port_args)
        template_manager = None

    # Wait for the model to finish loading
    scheduler_infos = []
    for i in range(len(scheduler_pipe_readers)):
        try:
            data = scheduler_pipe_readers[i].recv()
        except EOFError:
            logger.error(
                f"Rank {i} scheduler is dead. Please check if there are relevant logs."
            )
            scheduler_procs[i].join()
            logger.error(f"Exit code: {scheduler_procs[i].exitcode}")
            raise

        if data["status"] != "ready":
            raise RuntimeError(
                "Initialization failed. Please see the error messages above."
            )
        scheduler_infos.append(data)

    # Assume all schedulers have the same scheduler_info
    scheduler_info = scheduler_infos[0]
    tokenizer_manager.max_req_input_len = scheduler_info["max_req_input_len"]

    # *Simualtion Profiler
    print(f'Controller sending perf...')
    duration = Profiler.profile_time_host(start_time)
    simulator_controller.send_perf(
        {"_launch_subprocesses": duration}
    )

    return tokenizer_manager, template_manager, scheduler_info, port_args
