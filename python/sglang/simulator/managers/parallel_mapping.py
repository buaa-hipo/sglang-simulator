from sglang.srt.server_args import ServerArgs
from sglang.simulator.srt.simulator_args import SimulatorArgs


def TP_mapping(
    server_args: ServerArgs,
    simulator_args: SimulatorArgs,
    tp_rank: int,
):
    tp_rank_per_gpu = server_args.tp_size // simulator_args.gpu_size_actual
    tp_rank_mapped = tp_rank // tp_rank_per_gpu
    return tp_rank_mapped


def EP_mapping(
    server_args: ServerArgs,
    simulator_args: SimulatorArgs,
    moe_ep_rank: int,
):
    pass


def PD_DISAGG_mapping():
    pass


def gpu_id_mapping(
    server_args: ServerArgs,
    simulator_args: SimulatorArgs,
    tp_rank_mapped: int,
):
    nnodes_per_tp_group = server_args.nnodes
    tp_size_per_node = simulator_args.gpu_size_actual // nnodes_per_tp_group
    gpu_id_mapped = (
        simulator_args.base_gpu_id_actual
        + (tp_rank_mapped % tp_size_per_node) * simulator_args.gpu_id_step_actual
    )
    return gpu_id_mapped
