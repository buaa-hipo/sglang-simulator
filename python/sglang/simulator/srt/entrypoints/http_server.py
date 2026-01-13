import multiprocessing
from typing import Callable, Optional
import uvicorn
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    add_api_key_middleware,
    add_prometheus_track_response_middleware,
    set_uvicorn_logging_configs,
)
from sglang.srt.managers.multi_tokenizer_mixin import (
    monkey_patch_uvicorn_multiprocessing,
    write_data_for_multi_tokenizer,
)
from sglang.srt.entrypoints.http_server import (
    set_global_state,
    _GlobalState,
    app,
    _global_state,
)
from sglang.simulator.srt.entrypoints.engine import _launch_subprocesses


def launch_server(
    server_args: ServerArgs,
    pipe_finish_writer: Optional[multiprocessing.connection.Connection] = None,
    launch_callback: Optional[Callable[[], None]] = None,
):
    """
    Launch SRT (SGLang Runtime) Server.

    The SRT server consists of an HTTP server and an SRT engine.

    - HTTP server: A FastAPI server that routes requests to the engine.
    - The engine consists of three components:
        1. TokenizerManager: Tokenizes the requests and sends them to the scheduler.
        2. Scheduler (subprocess): Receives requests from the Tokenizer Manager, schedules batches, forwards them, and sends the output tokens to the Detokenizer Manager.
        3. DetokenizerManager (subprocess): Detokenizes the output tokens and sends the result back to the Tokenizer Manager.

    Note:
    1. The HTTP server, Engine, and TokenizerManager all run in the main process.
    2. Inter-process communication is done through IPC (each process uses a different port) via the ZMQ library.
    """
    tokenizer_manager, template_manager, scheduler_info, port_args = (
        _launch_subprocesses(server_args=server_args)
    )

    set_global_state(
        _GlobalState(
            tokenizer_manager=tokenizer_manager,
            template_manager=template_manager,
            scheduler_info=scheduler_info,
        )
    )

    if server_args.enable_metrics:
        add_prometheus_track_response_middleware(app)

    # Pass additional arguments to the lifespan function.
    # They will be used for additional initialization setups.
    if server_args.tokenizer_worker_num == 1:
        # If it is single tokenizer mode, we can pass the arguments by attributes of the app object.
        app.is_single_tokenizer_mode = True
        app.server_args = server_args
        app.warmup_thread_args = (
            server_args,
            pipe_finish_writer,
            launch_callback,
        )

        # Add api key authorization
        # This is only supported in single tokenizer mode.
        if server_args.api_key:
            add_api_key_middleware(app, server_args.api_key)
    else:
        # If it is multi-tokenizer mode, we need to write the arguments to shared memory
        # for other worker processes to read.
        app.is_single_tokenizer_mode = False
        multi_tokenizer_args_shm = write_data_for_multi_tokenizer(
            port_args, server_args, scheduler_info
        )

    try:
        # Update logging configs
        set_uvicorn_logging_configs()

        # Listen for HTTP requests
        if server_args.tokenizer_worker_num == 1:
            uvicorn.run(
                app,
                host=server_args.host,
                port=server_args.port,
                root_path=server_args.fastapi_root_path,
                log_level=server_args.log_level_http or server_args.log_level,
                timeout_keep_alive=5,
                loop="uvloop",
            )
        else:
            from uvicorn.config import LOGGING_CONFIG

            LOGGING_CONFIG["loggers"]["sglang.srt.entrypoints.http_server"] = {
                "handlers": ["default"],
                "level": "INFO",
                "propagate": False,
            }
            monkey_patch_uvicorn_multiprocessing()

            uvicorn.run(
                "sglang.srt.entrypoints.http_server:app",
                host=server_args.host,
                port=server_args.port,
                root_path=server_args.fastapi_root_path,
                log_level=server_args.log_level_http or server_args.log_level,
                timeout_keep_alive=5,
                loop="uvloop",
                workers=server_args.tokenizer_worker_num,
            )
    finally:
        if server_args.tokenizer_worker_num > 1:
            multi_tokenizer_args_shm.unlink()
            _global_state.tokenizer_manager.socket_mapping.clear_all_sockets()
