import setproctitle
import faulthandler
import psutil
import logging
import signal
import zmq
import time
from sglang.utils import get_exception_traceback
from sglang.srt.managers.scheduler import IdleSleeper
from sglang.srt.utils import (
    get_zmq_socket,
    kill_itself_when_parent_died,
)
from typing import Optional
import os
import json
from pathlib import Path


logger = logging.getLogger(__name__)


class Profiler:
    def __init__(
        self,
        controller_push_ipc_name: str,
        controller_pull_ipc_name: str,
        sleep_on_idle: bool = False,
        output_file: Optional[str] = None,
    ):
        context = zmq.Context()
        self.recv_from_controller = get_zmq_socket(
            context, zmq.PULL, controller_push_ipc_name, True
        )
        self.send_to_controller = get_zmq_socket(
            context, zmq.PUSH, controller_pull_ipc_name, False
        )

        if sleep_on_idle:
            self.idle_sleeper = IdleSleeper(
                [self.recv_from_controller]
            )
        else:
            self.idle_sleeper = None

        self.output_file = output_file
        self._fp = None
        if self.output_file is not None:
            parent = os.path.dirname(self.output_file)
            if parent:
                os.makedirs(parent, exist_ok=True)
            # os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            self._fp = open(self.output_file, "a", buffering=1)
    
    def recv_perf(self):
        try:
            recv_perf = self.recv_from_controller.recv_pyobj()
        except zmq.ZMQError:
            traceback = get_exception_traceback()
            logger.error(f"Profiler hit an exception: {traceback}")
            recv_perf = None
        return recv_perf

    def write_perf(self, perf: dict):
        if self._fp is None:
            return

        record = {**perf}
        self._fp.write(json.dumps(record) + "\n")    

    def event_loop_normal(self):
        while True:
            perf = self.recv_perf()
            if perf is not None:
                logger.info(f"Profiler received perf: {str(perf)}")
                self.write_perf(perf)
            else:
                self.maybe_sleep_on_idle()

    def maybe_sleep_on_idle(self):
        if self.idle_sleeper is not None:
            self.idle_sleeper.maybe_sleep()

    @staticmethod
    def profile_time_host(timer: float = None):
        if timer is None:
            return time.perf_counter()
        duration = time.perf_counter() - timer  # seconds
        return duration

PROFILER_DIR = os.getenv("PROFILER_DIR", "/tmp")

def run_profiler(
    controller_push_ipc_name: str,
    controller_pull_ipc_name: str,
    output_dir: Optional[str] = None,
):

    if output_dir is None:
        output_dir = PROFILER_DIR
    output_dir = Path(os.path.abspath(os.path.normpath(output_dir))) / str(int(time.time()))
    # output_dir = Path(PROFILER_DIR).resolve() / f"simulator_profile{int(time.time())}"
    output_dir.mkdir(parents=True, exist_ok=True)

    #output_file = output_dir / f"profile_profiler.jsonl"
    #output_dir = os.path.abspath(os.path.normpath(output_dir))
    output_file = os.path.join(
        output_dir,
        f"profile_profiler.jsonl"
    )

    # Config the process
    setproctitle.setproctitle("sglang::simulator::profiler")
    faulthandler.enable()
    kill_itself_when_parent_died()
    parent_process = psutil.Process().parent()

    # Create a profiler and run the event loop
    try:
        profiler = Profiler(
            controller_push_ipc_name,
            controller_pull_ipc_name,
            output_file=output_file,
        )
        if profiler._fp:
            profiler._fp.close()
        profiler.event_loop_normal()

    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Profiler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
    
    finally:
        if hasattr(profiler, "_fp") and profiler._fp:
            profiler._fp.close()

