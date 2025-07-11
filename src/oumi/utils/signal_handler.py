# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import atexit
import os
import signal
import sys
import threading
from typing import Any, Callable, Optional

from oumi.cli.cli_utils import CONSOLE
from oumi.utils.logging import logger


class GracefulInterruptHandler:
    """Handles graceful shutdown of processes on interrupt signals."""

    def __init__(self):
        """Initialize the graceful interrupt handler."""
        self._shutdown_event = threading.Event()
        self._cleanup_functions: list[Callable[[], None]] = []
        self._child_processes: list[Any] = []
        self._original_handlers: dict[int, Any] = {}
        self._shutdown_initiated = False
        self._lock = threading.Lock()

    def register_cleanup_function(self, func: Callable[[], None]) -> None:
        """Registers a cleanup function to be called during shutdown."""
        with self._lock:
            self._cleanup_functions.append(func)

    def register_child_process(self, process: Any) -> None:
        """Registers a child process to be terminated during shutdown."""
        with self._lock:
            self._child_processes.append(process)

    def remove_child_process(self, process: Any) -> None:
        """Removes a child process from the shutdown list."""
        with self._lock:
            if process in self._child_processes:
                self._child_processes.remove(process)

    def install_signal_handlers(self) -> None:
        """Installs signal handlers for graceful shutdown."""
        signals_to_handle = [signal.SIGINT, signal.SIGTERM]

        # Only handle SIGQUIT on Unix-like systems
        if hasattr(signal, "SIGQUIT"):
            signals_to_handle.append(signal.SIGQUIT)

        for sig in signals_to_handle:
            self._original_handlers[sig] = signal.signal(sig, self._signal_handler)

        # Register cleanup at exit
        atexit.register(self._cleanup)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handles interrupt signals by initiating graceful shutdown."""
        signal_name = signal.Signals(signum).name

        with self._lock:
            if self._shutdown_initiated:
                logger.warning(f"Received {signal_name} during shutdown, forcing exit")
                os._exit(1)
                return

            self._shutdown_initiated = True

        CONSOLE.print(
            f"\n[yellow]Received {signal_name}, "
            f"initiating graceful shutdown...[/yellow]"
        )
        logger.info(f"Received {signal_name}, initiating graceful shutdown")

        # Set shutdown event to signal other threads
        self._shutdown_event.set()

        # Start cleanup in a separate thread to avoid blocking signal handler
        cleanup_thread = threading.Thread(target=self._cleanup, daemon=True)
        cleanup_thread.start()

        # Give cleanup thread time to work, then exit
        cleanup_thread.join(timeout=10.0)

        if cleanup_thread.is_alive():
            logger.warning("Cleanup taking too long, forcing exit")
            os._exit(1)

        sys.exit(128 + signum)

    def _cleanup(self) -> None:
        """Performs cleanup operations."""
        try:
            logger.info("Starting cleanup operations")

            # 1. Terminate child processes
            self._terminate_child_processes()

            # 2. Run custom cleanup functions
            self._run_cleanup_functions()

            # 3. Cleanup distributed training
            self._cleanup_distributed()

            # 4. Cleanup GPU resources
            self._cleanup_gpu_resources()

            logger.info("Cleanup operations completed successfully")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def _terminate_child_processes(self) -> None:
        """Terminates all registered child processes."""
        with self._lock:
            child_processes = self._child_processes.copy()

        for process in child_processes:
            try:
                if hasattr(process, "poll") and process.poll() is None:
                    logger.info(f"Terminating child process PID: {process.pid}")
                    process.terminate()

                    # Give process time to terminate gracefully
                    try:
                        process.wait(timeout=5.0)
                    except TimeoutError:
                        logger.warning(
                            f"Process {process.pid} didn't terminate, killing"
                        )
                        process.kill()
                        process.wait()

                elif hasattr(process, "is_alive") and process.is_alive():
                    logger.info(f"Terminating multiprocessing process: {process}")
                    process.terminate()
                    process.join(timeout=5.0)

                    if process.is_alive():
                        logger.warning(f"Process {process} didn't terminate, killing")
                        process.kill()
                        process.join()

            except Exception as e:
                logger.error(f"Error terminating process {process}: {e}")

    def _run_cleanup_functions(self) -> None:
        """Runs all registered cleanup functions."""
        with self._lock:
            cleanup_functions = self._cleanup_functions.copy()

        for func in cleanup_functions:
            try:
                logger.debug(f"Running cleanup function: {func.__name__}")
                func()
            except Exception as e:
                logger.error(f"Error in cleanup function {func.__name__}: {e}")

    def _cleanup_distributed(self) -> None:
        """Cleans up distributed training resources."""
        try:
            # Import here to avoid circular dependencies
            from oumi.core.distributed import cleanup_distributed, is_distributed

            if is_distributed():
                logger.info("Cleaning up distributed training resources")
                cleanup_distributed()
        except Exception as e:
            logger.error(f"Error cleaning up distributed resources: {e}")

    def _cleanup_gpu_resources(self) -> None:
        """Cleans up GPU resources."""
        try:
            # Import here to avoid circular dependencies
            from oumi.utils.torch_utils import device_cleanup

            logger.info("Cleaning up GPU resources")
            device_cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up GPU resources: {e}")

    def restore_signal_handlers(self) -> None:
        """Restores original signal handlers."""
        for sig, handler in self._original_handlers.items():
            signal.signal(sig, handler)
        self._original_handlers.clear()

    def is_shutdown_requested(self) -> bool:
        """Returns True if shutdown has been requested."""
        return self._shutdown_event.is_set()

    def wait_for_shutdown(self, timeout: Optional[float] = None) -> bool:
        """Waits for shutdown signal. Returns True if shutdown was requested."""
        return self._shutdown_event.wait(timeout=timeout)


# Global instance for easy access
_global_handler: Optional[GracefulInterruptHandler] = None


def get_global_interrupt_handler() -> GracefulInterruptHandler:
    """Gets the global interrupt handler instance."""
    global _global_handler
    if _global_handler is None:
        _global_handler = GracefulInterruptHandler()
    return _global_handler


def install_global_signal_handlers() -> None:
    """Installs global signal handlers."""
    handler = get_global_interrupt_handler()
    handler.install_signal_handlers()


def register_cleanup_function(func: Callable[[], None]) -> None:
    """Registers a cleanup function with the global handler."""
    handler = get_global_interrupt_handler()
    handler.register_cleanup_function(func)


def is_shutdown_requested() -> bool:
    """Returns True if shutdown has been requested."""
    handler = get_global_interrupt_handler()
    return handler.is_shutdown_requested()


def wait_for_shutdown(timeout: Optional[float] = None) -> bool:
    """Waits for shutdown signal. Returns True if shutdown was requested."""
    handler = get_global_interrupt_handler()
    return handler.wait_for_shutdown(timeout=timeout)
