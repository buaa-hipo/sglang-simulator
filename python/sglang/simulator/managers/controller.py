from sglang.srt.server_args import ServerArgs
from sglang.simulator.srt.simulator_args import SimulatorArgs
from sglang.simulator.managers.parallel_mapping import (
    TP_mapping,
    EP_mapping,
    PD_DISAGG_mapping,
    gpu_id_mapping,
)


class Controller:
    server_args: ServerArgs
    simulator_args: SimulatorArgs

    def __init__(
        self,
        server_args: ServerArgs,
        simulator_args: SimulatorArgs,
    ):
        self.server_args = server_args,
        self.simulator_args = simulator_args

        assert(self.server_args.pp_size == 1, "Simulator has not support PP!")

    def __repr__(self):
        return (
            f"simulator_args={self.simulator_args}"
        )

    def tp_mapping(self, tp_rank: int):
        return TP_mapping(
            self.server_args,
            self.simulator_args,
            tp_rank,
        )
    
    def ep_mapping(self, moe_ep_rank: int):
        return EP_mapping(
            self.server_args,
            self.simulator_args,
            moe_ep_rank,
        )

    def pd_disagg_mapping(self):
        # TODO
        pass

    def gpu_id_mapping(self, tp_rank_mapped: int):
        return gpu_id_mapping(
            self.server_args,
            self.simulator_args,
            tp_rank_mapped,
        )


_CONTROLLER: Controller = None


def get_simulation_controller() -> Controller:
    assert _CONTROLLER is not None, "controller is not initialized"
    return _CONTROLLER


def init_simulation_controller(
    server_args: ServerArgs,
    simulator_args: SimulatorArgs,
):
    global _CONTROLLER
    _CONTROLLER = Controller(
        server_args,
        simulator_args,
    )
