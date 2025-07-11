"""Test script for the graceful interrupt handler."""

import time
from threading import Thread

from oumi.utils.signal_handler import (
    install_global_signal_handlers,
    is_shutdown_requested,
    register_cleanup_function,
    wait_for_shutdown,
)


def test_basic_signal_handling():
    """Test basic signal handling functionality."""
    print("Testing basic signal handling...")

    # Install signal handlers
    install_global_signal_handlers()

    # Register a cleanup function
    cleanup_called = False

    def test_cleanup():
        nonlocal cleanup_called
        cleanup_called = True
        print("Cleanup function called!")

    register_cleanup_function(test_cleanup)

    print("Signal handlers installed. Press Ctrl+C to test graceful shutdown.")

    # Simulate work with periodic shutdown checks
    for i in range(100):
        if is_shutdown_requested():
            print("Shutdown requested, exiting gracefully")
            break
        time.sleep(0.1)
        if i % 10 == 0:
            print(f"Working... {i}/100")

    print("Test completed successfully!")


def test_multiprocessing_simulation():
    """Test simulated multiprocessing scenario."""
    print("Testing multiprocessing simulation...")

    install_global_signal_handlers()

    # Simulate multiple worker threads
    workers = []

    def worker(worker_id):
        """Simulate a worker process."""
        print(f"Worker {worker_id} started")
        while not is_shutdown_requested():
            time.sleep(0.5)
            print(f"Worker {worker_id} working...")
        print(f"Worker {worker_id} shutting down gracefully")

    # Start multiple workers
    for i in range(3):
        worker_thread = Thread(target=worker, args=(i,))
        worker_thread.start()
        workers.append(worker_thread)

    print("All workers started. Press Ctrl+C to test coordinated shutdown.")

    # Wait for shutdown or timeout
    shutdown_received = wait_for_shutdown(timeout=10)

    if shutdown_received:
        print("Shutdown signal received, waiting for workers to finish...")
        for worker_thread in workers:
            worker_thread.join(timeout=2)
        print("All workers shut down")
    else:
        print("Test completed without shutdown signal")
