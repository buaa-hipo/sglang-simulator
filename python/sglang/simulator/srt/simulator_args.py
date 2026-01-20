import dataclasses
import argparse
from typing import List


@dataclasses.dataclass
class SimulatorArgs:
    gpu_size_actual: int
    base_gpu_id_actual: int = 0
    gpu_id_step_actual: int = 1

    # Profiler
    sleep_on_idle: bool = False

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--gpu-size-actual",
            type=int,
            help="Actual number of GPUs/NPUs.",
            required=True,
        )
        parser.add_argument(
            "--base-gpu-id-actual",
            type=int,
            default=SimulatorArgs.base_gpu_id_actual,
            help="The base GPU ID to start allocating GPUs from. Useful when running multiple instances on the same machine.",
        )
        parser.add_argument(
            "--gpu-id-step-actual",
            type=int,
            default=SimulatorArgs.gpu_id_step_actual,
            help="The delta between consecutive GPU IDs that are used. For example, setting it to 2 will use GPU 0,2,4,...",
        )

        # Profiler
        parser.add_argument(
            "--sleep-on-idle",
            action="store_true",
            help="Reduce CPU usage when sglang is idle.",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})


def prepare_simulator_args(argv: List[str]) -> SimulatorArgs:
    # Import here to avoid circular imports
    from sglang.simulator.srt.server_args_config_parser import ConfigArgumentMerger

    parser = argparse.ArgumentParser()
    SimulatorArgs.add_cli_args(parser)
    raw_args, remaining_argv = parser.parse_known_args(argv)

    return SimulatorArgs.from_cli_args(raw_args), remaining_argv
