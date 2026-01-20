import tempfile
import multiprocessing as mp
import zmq
from typing import Any
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_zmq_socket
from sglang.simulator.srt.simulator_args import SimulatorArgs
from sglang.simulator.managers.parallel_mapping import (
    TP_mapping,
    EP_mapping,
    PD_DISAGG_mapping,
    gpu_id_mapping,
)
from sglang.simulator.profiler.profiler import run_profiler


class Controller:
    server_args: ServerArgs
    simulator_args: SimulatorArgs

    def __init__(
        self,
        server_args: ServerArgs,
        simulator_args: SimulatorArgs,
    ):
        self.server_args = server_args
        assert(self.server_args.pp_size == 1, "Simulator has not support PP!")
        self.simulator_args = simulator_args

        self._init_profiler()

    def __repr__(self):
        return (
            f"simulator_args={self.simulator_args}"
            f"profiler_ipc_name={self.profiler_ipc_name}"
        )

    def _init_profiler(self):
        self.profiler_push_ipc_name = f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
        self.profiler_pull_ipc_name = f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
        context = zmq.Context()
        self.send_to_profiler = get_zmq_socket(
            context, zmq.PUSH, self.profiler_push_ipc_name, False
        )
        self.recv_from_profiler = get_zmq_socket(
            context, zmq.PULL, self.profiler_pull_ipc_name, True
        )

        proc = mp.Process(
            target=run_profiler,
            args=(
                self.profiler_push_ipc_name,
                self.profiler_pull_ipc_name,
            ),
        )
        proc.start()

    def send_perf(self, perf: Any):
        #self.send_to_profiler.send_pyobj(perf)
        try:
            self.send_to_profiler.send_pyobj(perf, zmq.NOBLOCK)
        except zmq.Again:
            print("Profiler not ready, drop perf")

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

    def get_ccl_backend(self) -> str:
        return self.simulator_args.ccl_backend


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
