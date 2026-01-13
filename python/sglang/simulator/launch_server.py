"""Launch the inference server."""

import asyncio
import os
import sys

from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree
from sglang.simulator.srt.simulator_args import prepare_simulator_args
from sglang.simulator.managers.controller import init_simulation_controller


def run_server(server_args):
    """Run the server based on server_args.grpc_mode."""
    if server_args.grpc_mode:
        raise Exception("grpc_mode is not implemented!")
        # from sglang.srt.entrypoints.grpc_server import serve_grpc

        # asyncio.run(serve_grpc(server_args))
    else:
        # Default mode: HTTP mode.
        # *Simulation
        from sglang.simulator.srt.entrypoints.http_server import launch_server

        launch_server(server_args)


if __name__ == "__main__":
    simulator_args, remaining_argv = prepare_simulator_args(sys.argv[1:])
    server_args = prepare_server_args(remaining_argv)
    init_simulation_controller()

    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
