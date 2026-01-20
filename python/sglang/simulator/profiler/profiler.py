import setproctitle
import faulthandler
import psutil
import logging
import signal
import zmq
import time
from sglang.utils import get_exception_traceback
from sglang.simulator.srt.managers.scheduler import IdleSleeper
from sglang.simulator.srt.utils import (
    get_zmq_socket,
    kill_itself_when_parent_died,
)


logger = logging.getLogger(__name__)


class Profiler:
    def __init__(
        self,
        controller_push_ipc_name: str,
        controller_pull_ipc_name: str,
        sleep_on_idle: bool = False,
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
    
    def recv_perf(self):
        try:
            recv_perf = self.recv_from_controller.recv_pyobj()
        except zmq.ZMQError:
            traceback = get_exception_traceback()
            logger.error(f"Profiler hit an exception: {traceback}")
            recv_perf = None
        return recv_perf

    def event_loop_normal(self):
        while True:
            perf = self.recv_perf()
            if perf is not None:
                logger.info(f"Profiler received perf: {str(perf)}")
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


def run_profiler(
    controller_push_ipc_name: str,
    controller_pull_ipc_name: str,
):
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
        )
        profiler.event_loop_normal()

    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Profiler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
