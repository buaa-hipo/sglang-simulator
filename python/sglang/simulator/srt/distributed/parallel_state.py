import contextlib
import gc
import logging
import os
import pickle
import weakref
from collections import namedtuple
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import timedelta
from multiprocessing import shared_memory
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from unittest.mock import patch

import torch
import torch.distributed
from sglang.srt.distributed import GroupCoordinator
from sglang.srt.environ import envs
from sglang.srt.utils import (
    direct_register_custom_op,
    get_bool_env_var,
    get_current_device_stream_fast,
    get_int_env_var,
    get_local_ip_auto,
    is_cpu,
    is_cuda_alike,
    is_hip,
    is_npu,
    is_shm_available,
    is_xpu,
    supports_custom_op,
)

_is_npu = is_npu()
_is_cpu = is_cpu()
_is_xpu = is_xpu()
_supports_custom_op = supports_custom_op()

from sglang.simulator.srt.distributed.device_communicators.npu_communicator import NpuCommunicatorSim


TensorMetadata = namedtuple("TensorMetadata", ["device", "dtype", "size"])

# use int value instead of ReduceOp.SUM to support torch compile
REDUCE_OP_SUM = int(torch.distributed.ReduceOp.SUM)


@dataclass
class GraphCaptureContext:
    stream: torch.get_device_module().Stream


@dataclass
class P2PWork:
    work: Optional[torch.distributed.Work]
    payload: Optional[torch.Tensor]


def _split_tensor_dict(
    tensor_dict: Dict[str, Union[torch.Tensor, Any]]
) -> Tuple[List[Tuple[str, Any]], List[torch.Tensor]]:
    """Split the tensor dictionary into two parts:
    1. A list of (key, value) pairs. If the value is a tensor, it is replaced
         by its metadata.
    2. A list of tensors.
    """
    metadata_list: List[Tuple[str, Any]] = []
    tensor_list: List[torch.Tensor] = []
    for key, value in tensor_dict.items():
        if isinstance(value, torch.Tensor):
            # Note: we cannot use `value.device` here,
            # because it contains not only the device type but also the device
            # index (e.g. "cuda:0"). We only need the device type.
            # receiving side will set the device index.
            device = value.device.type
            metadata_list.append(
                (key, TensorMetadata(device, value.dtype, value.size()))
            )
            tensor_list.append(value)
        else:
            metadata_list.append((key, value))
    return metadata_list, tensor_list


_group_name_counter: Dict[str, int] = {}


def _get_unique_name(name: str) -> str:
    """Get a unique name for the group.
    Example:
    _get_unique_name("tp") -> "tp:0"
    _get_unique_name("tp") -> "tp:1"
    """
    if name not in _group_name_counter:
        _group_name_counter[name] = 0
    newname = f"{name}:{_group_name_counter[name]}"
    _group_name_counter[name] += 1
    return newname


_groups: Dict[str, Callable[[], Optional["GroupCoordinatorSim"]]] = {}


def _register_group(group: "GroupCoordinatorSim") -> None:
    _groups[group.unique_name] = weakref.ref(group)


if _supports_custom_op:

    def inplace_all_reduce(tensor: torch.Tensor, group_name: str) -> None:
        assert group_name in _groups, f"Group {group_name} is not found."
        group = _groups[group_name]()
        if group is None:
            raise ValueError(f"Group {group_name} is destroyed.")
        group._all_reduce_in_place(tensor)

    def inplace_all_reduce_fake(tensor: torch.Tensor, group_name: str) -> None:
        return

    direct_register_custom_op(
        op_name="inplace_all_reduce",
        op_func=inplace_all_reduce,
        mutates_args=["tensor"],
        fake_impl=inplace_all_reduce_fake,
    )

    def outplace_all_reduce(
        tensor: torch.Tensor, group_name: str, outplace_all_reduce_method: str
    ) -> torch.Tensor:
        assert group_name in _groups, f"Group {group_name} is not found."
        group = _groups[group_name]()
        if group is None:
            raise ValueError(f"Group {group_name} is destroyed.")
        return group._all_reduce_out_place(tensor, outplace_all_reduce_method)

    def outplace_all_reduce_fake(
        tensor: torch.Tensor, group_name: str, outplace_all_reduce_method: str
    ) -> torch.Tensor:
        return torch.empty_like(tensor)

    direct_register_custom_op(
        op_name="outplace_all_reduce",
        op_func=outplace_all_reduce,
        mutates_args=[],
        fake_impl=outplace_all_reduce_fake,
    )

    def reg_all_gather_into_tensor(
        output: torch.Tensor, input: torch.Tensor, group_name: str
    ) -> None:
        assert group_name in _groups, f"Group {group_name} is not found."
        group = _groups[group_name]()
        if group is None:
            raise ValueError(f"Group {group_name} is destroyed.")
        group._all_gather_into_tensor(output, input)

    def reg_all_gather_into_tensor_fake(
        output: torch.Tensor, input: torch.Tensor, group_name: str
    ) -> None:
        pass

    direct_register_custom_op(
        op_name="reg_all_gather_into_tensor",
        op_func=reg_all_gather_into_tensor,
        mutates_args=["output"],
        fake_impl=reg_all_gather_into_tensor_fake,
    )

    def reg_reduce_scatter_tensor(
        output: torch.Tensor, input: torch.Tensor, group_name: str
    ) -> None:
        assert group_name in _groups, f"Group {group_name} is not found."
        group = _groups[group_name]()
        if group is None:
            raise ValueError(f"Group {group_name} is destroyed.")
        group._reduce_scatter_tensor(output, input)

    def reg_reduce_scatter_tensor_fake(
        output: torch.Tensor, input: torch.Tensor, group_name: str
    ) -> None:
        pass

    direct_register_custom_op(
        op_name="reg_reduce_scatter_tensor",
        op_func=reg_reduce_scatter_tensor,
        mutates_args=["output"],
        fake_impl=reg_reduce_scatter_tensor_fake,
    )


class GroupCoordinatorSim(GroupCoordinator):
    def __init__(
        self,
        group_ranks: List[List[int]],
        local_rank: int,
        torch_distributed_backend: Union[str, Backend],
        use_pynccl: bool,
        use_pymscclpp: bool,
        use_custom_allreduce: bool,
        use_torch_symm_mem_all_reduce: bool,
        use_hpu_communicator: bool,
        use_xpu_communicator: bool,
        use_npu_communicator: bool,
        use_message_queue_broadcaster: bool = False,
        group_name: Optional[str] = None,
        pynccl_use_current_stream: bool = False,
        torch_compile: Optional[bool] = None,
        gloo_timeout: timedelta = timedelta(seconds=120 * 60),
    ):
        super().__init__(group_ranks, local_rank, torch_distributed_backend, use_pynccl, use_pymscclpp, use_custom_allreduce, use_torch_symm_mem_all_reduce, use_hpu_communicator, use_xpu_communicator, use_npu_communicator, use_message_queue_broadcaster, group_name, pynccl_use_current_stream, torch_compile, gloo_timeout)
        self.npu_communicator: Optional[NpuCommunicatorSim] = None
        if use_npu_communicator and self.world_size > 1:
            self.npu_communicator = NpuCommunicatorSim(group=self.device_group)

_WORLD: Optional[GroupCoordinatorSim] = None


def get_world_group() -> GroupCoordinatorSim:
    assert _WORLD is not None, "world group is not initialized"
    return _WORLD


def init_world_group(
    ranks: List[int], local_rank: int, backend: str
) -> GroupCoordinatorSim:
    return GroupCoordinatorSim(
        group_ranks=[ranks],
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_pynccl=False,
        use_pymscclpp=False,
        use_custom_allreduce=False,
        use_torch_symm_mem_all_reduce=False,
        use_hpu_communicator=False,
        use_xpu_communicator=False,
        use_npu_communicator=False,
        group_name="world",
    )


def init_model_parallel_group(
    group_ranks: List[List[int]],
    local_rank: int,
    backend: str,
    use_custom_allreduce: Optional[bool] = None,
    use_message_queue_broadcaster: bool = False,
    group_name: Optional[str] = None,
    use_mscclpp_allreduce: Optional[bool] = None,
    pynccl_use_current_stream: bool = True,
    use_torch_symm_mem_allreduce: Optional[bool] = None,
    torch_compile: Optional[bool] = None,
) -> GroupCoordinatorSim:
    if use_custom_allreduce is None:
        use_custom_allreduce = _ENABLE_CUSTOM_ALL_REDUCE
    if use_mscclpp_allreduce is None:
        use_mscclpp_allreduce = _ENABLE_MSCCLPP_ALL_REDUCE
    if use_torch_symm_mem_allreduce is None:
        use_torch_symm_mem_allreduce = _ENABLE_TORCH_SYMM_MEM_ALL_REDUCE
    return GroupCoordinatorSim(
        group_ranks=group_ranks,
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_pynccl=not (_is_npu or _is_xpu),
        use_pymscclpp=use_mscclpp_allreduce,
        use_custom_allreduce=use_custom_allreduce,
        use_torch_symm_mem_all_reduce=use_torch_symm_mem_allreduce,
        use_hpu_communicator=True,
        use_xpu_communicator=True,
        use_npu_communicator=True,
        use_message_queue_broadcaster=use_message_queue_broadcaster,
        group_name=group_name,
        pynccl_use_current_stream=pynccl_use_current_stream,
        torch_compile=torch_compile,
    )


_TP: Optional[GroupCoordinatorSim] = None

# duplicate GroupCoordinatorSim for prefill in PD-Multiplexing
_PDMUX_PREFILL_TP_GROUP: Optional[GroupCoordinatorSim] = None

_ENABLE_PDMUX_P_TP: bool = False


def set_pdmux_status(enable_prefill_multiplexing: bool):
    global _ENABLE_PDMUX_P_TP
    _ENABLE_PDMUX_P_TP = enable_prefill_multiplexing


def get_tp_group() -> GroupCoordinatorSim:
    if _ENABLE_PDMUX_P_TP:
        assert (
            _PDMUX_PREFILL_TP_GROUP is not None
        ), "tensor model parallel group for PD-Multiplexing Prefill is not initialized"
        return _PDMUX_PREFILL_TP_GROUP
    assert _TP is not None, "tensor model parallel group is not initialized"
    return _TP


_MOE_EP: Optional[GroupCoordinatorSim] = None
_MOE_TP: Optional[GroupCoordinatorSim] = None


def get_moe_ep_group() -> GroupCoordinatorSim:
    assert _MOE_EP is not None, "expert model parallel group is not initialized"
    return _MOE_EP


def get_moe_tp_group() -> GroupCoordinatorSim:
    assert _MOE_TP is not None, "expert model parallel group is not initialized"
    return _MOE_TP


# kept for backward compatibility
get_tensor_model_parallel_group = get_tp_group

_PP: Optional[GroupCoordinatorSim] = None


def get_pp_group() -> GroupCoordinatorSim:
    assert _PP is not None, "pipeline model parallel group is not initialized"
    return _PP


# kept for backward compatibility
get_pipeline_model_parallel_group = get_pp_group


@contextmanager
def graph_capture(stream: Optional[torch.cuda.Stream] = None):
    """
    `graph_capture` is a context manager which should surround the code that
    is capturing the CUDA graph. Its main purpose is to ensure that the
    some operations will be run after the graph is captured, before the graph
    is replayed. It returns a `GraphCaptureContext` object which contains the
    necessary data for the graph capture. Currently, it only contains the
    stream that the graph capture is running on. This stream is set to the
    current CUDA stream when the context manager is entered and reset to the
    default stream when the context manager is exited. This is to ensure that
    the graph capture is running on a separate stream from the default stream,
    in order to explicitly distinguish the kernels to capture
    from other kernels possibly launched on background in the default stream.
    """
    with get_tp_group().graph_capture(
        stream=stream
    ) as context, get_pp_group().graph_capture(context):
        yield context


logger = logging.getLogger(__name__)

_ENABLE_CUSTOM_ALL_REDUCE = True
_ENABLE_MSCCLPP_ALL_REDUCE = False
_ENABLE_TORCH_SYMM_MEM_ALL_REDUCE = False


def set_custom_all_reduce(enable: bool):
    global _ENABLE_CUSTOM_ALL_REDUCE
    _ENABLE_CUSTOM_ALL_REDUCE = enable


def set_mscclpp_all_reduce(enable: bool):
    global _ENABLE_MSCCLPP_ALL_REDUCE
    _ENABLE_MSCCLPP_ALL_REDUCE = enable


def set_torch_symm_mem_all_reduce(enable: bool):
    global _ENABLE_TORCH_SYMM_MEM_ALL_REDUCE
    _ENABLE_TORCH_SYMM_MEM_ALL_REDUCE = enable


def init_distributed_environment(
    world_size: int = -1,
    rank: int = -1,
    distributed_init_method: str = "env://",
    local_rank: int = -1,
    backend: str = "nccl",
    timeout: Optional[int] = None,
):
    logger.debug(
        "world_size=%d rank=%d local_rank=%d " "distributed_init_method=%s backend=%s",
        world_size,
        rank,
        local_rank,
        distributed_init_method,
        backend,
    )
    if "mooncake" in backend:
        try:
            from mooncake import ep as mooncake_ep
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run SGLang with Mooncake Backend."
            ) from e
        mooncake_ep.set_host_ip(get_local_ip_auto())

    if not torch.distributed.is_initialized():
        assert distributed_init_method is not None, (
            "distributed_init_method must be provided when initializing "
            "distributed environment"
        )
        if timeout is not None:
            assert isinstance(timeout, (int)), "timeout must be a number"
            assert timeout > 0, "timeout must be positive"
            timeout = timedelta(seconds=timeout)

        # this backend is used for WORLD
        torch.distributed.init_process_group(
            backend=backend,
            init_method=distributed_init_method,
            world_size=world_size,
            rank=rank,
            timeout=timeout,
        )

    # set the local rank
    # local_rank is not available in torch ProcessGroup,
    # see https://github.com/pytorch/pytorch/issues/122816
    if local_rank == -1:
        # local rank not set, this usually happens in single-node
        # setting, where we can use rank as local rank
        if distributed_init_method == "env://":
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        else:
            local_rank = rank
    global _WORLD
    if _WORLD is None:
        ranks = list(range(torch.distributed.get_world_size()))
        _WORLD = init_world_group(ranks, local_rank, backend)
    else:
        assert (
            _WORLD.world_size == torch.distributed.get_world_size()
        ), "world group already initialized with a different world size"


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    backend: Optional[str] = None,
    duplicate_tp_group: bool = False,
    torch_compile: Optional[bool] = None,
) -> None:
    """
    Initialize model parallel groups.

    Arguments:
        tensor_model_parallel_size: number of GPUs used for tensor model
            parallelism.
        pipeline_model_parallel_size: number of GPUs used for pipeline model
            parallelism.

    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 4 tensor model-parallel groups and 2 pipeline model-parallel groups:
        4 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 pipeline model-parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    if world_size != tensor_model_parallel_size * pipeline_model_parallel_size:
        raise RuntimeError(
            f"world_size ({world_size}) is not equal to "
            f"tensor_model_parallel_size ({tensor_model_parallel_size}) x "
            f"pipeline_model_parallel_size ({pipeline_model_parallel_size})"
        )

    # Build the tensor model-parallel groups.
    num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
    global _TP
    assert _TP is None, "tensor model parallel group is already initialized"
    group_ranks = []
    for i in range(num_tensor_model_parallel_groups):
        ranks = list(
            range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
        )
        group_ranks.append(ranks)

    # message queue broadcaster is only used in tensor model parallel group
    _TP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=get_bool_env_var(
            "SGLANG_USE_MESSAGE_QUEUE_BROADCASTER", "true"
        ),
        group_name="tp",
        pynccl_use_current_stream=duplicate_tp_group,
        torch_compile=torch_compile,
    )

    if duplicate_tp_group:
        global _PDMUX_PREFILL_TP_GROUP
        assert (
            _PDMUX_PREFILL_TP_GROUP is None
        ), "tensor model parallel group for PD-Multiplexing Prefill is already initialized"
        _PDMUX_PREFILL_TP_GROUP = init_model_parallel_group(
            group_ranks,
            get_world_group().local_rank,
            backend,
            use_message_queue_broadcaster=get_bool_env_var(
                "SGLANG_USE_MESSAGE_QUEUE_BROADCASTER", "true"
            ),
            group_name="pdmux_prefill_tp",
            pynccl_use_current_stream=True,
            torch_compile=torch_compile,
        )
        if _TP.pynccl_comm:
            _TP.pynccl_comm.disabled = False
            _PDMUX_PREFILL_TP_GROUP.pynccl_comm.disabled = False

    moe_ep_size = expert_model_parallel_size
    moe_tp_size = tensor_model_parallel_size // moe_ep_size

    global _MOE_EP
    assert _MOE_EP is None, "expert model parallel group is already initialized"
    if moe_ep_size == tensor_model_parallel_size:
        _MOE_EP = _TP
    else:
        # TODO(ch-wan): use split_group to save memory
        group_ranks = []
        for i in range(num_tensor_model_parallel_groups):
            for j in range(moe_tp_size):
                st = i * tensor_model_parallel_size + j
                en = (i + 1) * tensor_model_parallel_size + j
                ranks = list(range(st, en, moe_tp_size))
                group_ranks.append(ranks)
        _MOE_EP = init_model_parallel_group(
            group_ranks,
            get_world_group().local_rank,
            backend,
            group_name="moe_ep",
        )

    global _MOE_TP
    assert _MOE_TP is None, "expert model parallel group is already initialized"
    if moe_tp_size == tensor_model_parallel_size:
        _MOE_TP = _TP
    else:
        # TODO(ch-wan): use split_group to save memory
        group_ranks = []
        for i in range(num_tensor_model_parallel_groups):
            for j in range(moe_ep_size):
                st = i * tensor_model_parallel_size + j * moe_tp_size
                en = i * tensor_model_parallel_size + (j + 1) * moe_tp_size
                ranks = list(range(st, en))
                group_ranks.append(ranks)
        _MOE_TP = init_model_parallel_group(
            group_ranks,
            get_world_group().local_rank,
            backend,
            group_name="moe_tp",
        )

    # Build the pipeline model-parallel groups.
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
    global _PP
    assert _PP is None, "pipeline model parallel group is already initialized"
    group_ranks = []
    for i in range(num_pipeline_model_parallel_groups):
        ranks = list(range(i, world_size, num_pipeline_model_parallel_groups))
        group_ranks.append(ranks)
    # pipeline parallel does not need custom allreduce
    _PP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_custom_allreduce=False,
        group_name="pp",
    )


def ensure_model_parallel_initialized(
    tensor_model_parallel_size: int,
    expert_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    backend: Optional[str] = None,
) -> None:
    """Helper to initialize model parallel groups if they are not initialized,
    or ensure tensor-parallel and pipeline-parallel sizes are equal to expected
    values if the model parallel groups are initialized.
    """
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)
    if not model_parallel_is_initialized():
        initialize_model_parallel(
            tensor_model_parallel_size,
            expert_model_parallel_size,
            pipeline_model_parallel_size,
            backend,
        )
        return

    assert get_tensor_model_parallel_world_size() == tensor_model_parallel_size, (
        "tensor parallel group already initialized, but of unexpected size: "
        f"{get_tensor_model_parallel_world_size()=} vs. "
        f"{tensor_model_parallel_size=}"
    )
    pp_world_size = get_pp_group().world_size
    assert pp_world_size == pipeline_model_parallel_size, (
        "pipeline parallel group already initialized, but of unexpected size: "
        f"{pp_world_size=} vs. "
        f"{pipeline_model_parallel_size=}"
    )


def model_parallel_is_initialized():
    """Check if tensor and pipeline parallel groups are initialized."""
    return _TP is not None and _PP is not None


_TP_STATE_PATCHED = False


@contextmanager
def patch_tensor_parallel_group(tp_group: GroupCoordinatorSim):
    """Patch the tp group temporarily until this function ends.

    This method is for draft workers of speculative decoding to run draft model
    with different tp degree from that of target model workers.

    Args:
        tp_group (GroupCoordinatorSim): the tp group coordinator
    """
    global _TP_STATE_PATCHED
    assert not _TP_STATE_PATCHED, "Should not call when it's already patched"

    _TP_STATE_PATCHED = True
    old_tp_group = get_tp_group()
    global _TP
    _TP = tp_group
    try:
        yield
    finally:
        # restore the original state
        _TP_STATE_PATCHED = False
        _TP = old_tp_group


def get_world_size():
    """Return world size for the world group."""
    return get_world_group().world_size


def get_world_rank():
    """Return my rank for the world group."""
    return get_world_group().rank_in_group


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    return get_tp_group().world_size


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    return get_tp_group().rank_in_group


def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    return get_pp_group().world_size


def get_pipeline_model_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    return get_pp_group().rank_in_group


def get_moe_expert_parallel_world_size():
    """Return world size for the moe expert parallel group."""
    return get_moe_ep_group().world_size


def get_moe_expert_parallel_rank():
    """Return my rank for the moe expert parallel group."""
    return get_moe_ep_group().rank_in_group


def get_moe_tensor_parallel_world_size():
    """Return world size for the moe tensor parallel group."""
    return get_moe_tp_group().world_size


def get_moe_tensor_parallel_rank():
    """Return my rank for the moe tensor parallel group."""
    return get_moe_tp_group().rank_in_group


def destroy_model_parallel():
    """Set the groups to none and destroy them."""
    global _TP
    if _TP:
        _TP.destroy()
    _TP = None

    global _PP
    if _PP:
        _PP.destroy()
    _PP = None

    global _PDMUX_PREFILL_TP_GROUP
    if _PDMUX_PREFILL_TP_GROUP:  # type: ignore[union-attr]
        _PDMUX_PREFILL_TP_GROUP.destroy()
    _PDMUX_PREFILL_TP_GROUP = None


def destroy_distributed_environment():
    global _WORLD
    if _WORLD:
        _WORLD.destroy()
    _WORLD = None
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def cleanup_dist_env_and_memory(shutdown_ray: bool = False):
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    if shutdown_ray:
        import ray  # Lazy import Ray

        ray.shutdown()
    gc.collect()
    if not _is_cpu:
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch._C, "_host_emptyCache"):
                torch._C._host_emptyCache()
            else:
                logger.warning(
                    "torch._C._host_emptyCache() only available in Pytorch >=2.5"
                )
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
        elif hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.empty_cache()


def in_the_same_node_as(pg: ProcessGroup, source_rank: int = 0) -> List[bool]:
    """
    This is a collective operation that returns if each rank is in the same node
    as the source rank. It tests if processes are attached to the same
    memory system (shared access to shared memory).
    """
    assert (
        torch.distributed.get_backend(pg) != torch.distributed.Backend.NCCL
    ), "in_the_same_node_as should be tested with a non-NCCL group."
    # local rank inside the group
    rank = torch.distributed.get_rank(group=pg)
    world_size = torch.distributed.get_world_size(group=pg)

    # local tensor in each process to store the result
    is_in_the_same_node = torch.tensor([0] * world_size, dtype=torch.int32)

    # global ranks of the processes in the group
    ranks = torch.distributed.get_process_group_ranks(pg)

    magic_message = b"magic_message"
    shm = None

    try:
        with contextlib.suppress(OSError):
            if rank == source_rank:
                # create a shared memory segment
                shm = shared_memory.SharedMemory(create=True, size=128)
                shm.buf[: len(magic_message)] = magic_message
                torch.distributed.broadcast_object_list(
                    [shm.name], src=ranks[source_rank], group=pg
                )
                is_in_the_same_node[rank] = 1
            else:
                # try to open the shared memory segment
                recv = [None]
                torch.distributed.broadcast_object_list(
                    recv, src=ranks[source_rank], group=pg
                )
                name = recv[0]
                # fix to https://stackoverflow.com/q/62748654/9191338
                # Python incorrectly tracks shared memory even if it is not
                # created by the process. The following patch is a workaround.
                with patch(
                    "multiprocessing.resource_tracker.register",
                    lambda *args, **kwargs: None,
                ):
                    shm = shared_memory.SharedMemory(name=name)
                if shm.buf[: len(magic_message)] == magic_message:
                    is_in_the_same_node[rank] = 1
    except Exception as e:
        logger.error("Error ignored in is_in_the_same_node: %s", e)
    finally:
        if shm:
            shm.close()

    torch.distributed.barrier(group=pg)

    # clean up the shared memory segment
    with contextlib.suppress(OSError):
        if rank == source_rank and shm:
            shm.unlink()
    torch.distributed.all_reduce(is_in_the_same_node, group=pg)

    return [x == 1 for x in is_in_the_same_node.tolist()]


vllm_get_pp_group = None
vllm_get_tp_group = None
vllm_get_world_group = None


def monkey_patch_vllm_parallel_state(reverse: bool = False):
    try:
        import vllm.distributed.parallel_state as vllm_parrlel_state
    except ImportError:
        return

    global vllm_get_pp_group, vllm_get_tp_group, vllm_get_world_group
    if vllm_get_pp_group is None:
        vllm_get_pp_group = vllm_parrlel_state.get_pp_group
        vllm_get_tp_group = vllm_parrlel_state.get_tp_group
        vllm_get_world_group = vllm_parrlel_state.get_world_group
    if reverse:
        setattr(vllm_parrlel_state, "get_pp_group", vllm_get_pp_group)
        setattr(vllm_parrlel_state, "get_tp_group", vllm_get_tp_group)
        setattr(vllm_parrlel_state, "get_world_group", vllm_get_world_group)
    else:
        setattr(vllm_parrlel_state, "get_pp_group", get_pp_group)
        setattr(vllm_parrlel_state, "get_tp_group", get_tp_group)
        setattr(vllm_parrlel_state, "get_world_group", get_world_group)
