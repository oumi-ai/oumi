import argparse
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from oumi.core.distributed import (
    barrier,
    cleanup_distributed,
    get_device_rank_info,
    init_distributed,
    is_distributed,
    is_local_process_zero,
)
from oumi.models.mlp import MLPEncoder
from oumi.utils.logging import logger, update_logger_level


def benchmark_barrier(device_info, num_iterations: int = 1):
    """Benchmarks the time taken to execute a barrier operation."""
    start_time = time.perf_counter()
    sleep_time_seconds = 1
    for _ in range(num_iterations):
        barrier()
        time.sleep(sleep_time_seconds)
    end_time = time.perf_counter()

    total_time_seconds = end_time - start_time
    total_barrier_time_seconds = (
        total_time_seconds - sleep_time_seconds * num_iterations
    )
    avg_time = total_barrier_time_seconds / num_iterations

    if is_local_process_zero():
        logger.info(f"Average barrier time: {avg_time * 1000:.3f} ms")


def benchmark_ddp(device_info, num_iterations: int = 1):
    """Benchmarks the time taken to execute a forward and backward pass with DDP."""
    rank = device_info.local_rank
    device = f"cuda:{rank}"

    model = MLPEncoder().to(device)
    ddp_model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)

    forward_times = []
    backward_times = []

    for _ in range(num_iterations):
        input_tensor = torch.randint(high=768, size=(1000,)).to(rank)
        target_tensor = torch.randint(high=10, size=(1000,)).to(rank)

        # Forward pass
        forward_start = time.perf_counter()
        output = ddp_model(input_ids=input_tensor, labels=target_tensor)
        loss = output["loss"].sum()

        forward_end = time.perf_counter()
        forward_times.append(forward_end - forward_start)

        # Backward pass
        backward_start = time.perf_counter()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        backward_end = time.perf_counter()
        backward_times.append(backward_end - backward_start)

        torch.cuda.synchronize()

    avg_forward = sum(forward_times) / num_iterations
    avg_backward = sum(backward_times) / num_iterations

    if is_local_process_zero():
        logger.info(f"Average DDP forward time: {avg_forward * 1000:.3f} ms")
        logger.info(f"Average DDP backward time: {avg_backward * 1000:.3f} ms")


def benchmark_all_reduce(device_info, num_iterations: int = 1, tensor_size=1000000):
    """Benchmarks the time taken to execute an all-reduce operation."""
    tensor = torch.randn(tensor_size).to(device_info.local_rank)

    start_time = time.time()
    for _ in range(num_iterations):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / num_iterations

    if is_local_process_zero():
        logger.info(f"Average all_reduce time: {avg_time * 1000:.3f} ms")


def main(args):
    """Main function for the benchmark script."""
    if is_distributed():
        logger.info("Benchmarking in distributed mode...")
        init_distributed()
    else:
        logger.info("Running benchmark in single-process mode.")

    #
    # Run tests
    #
    device_info = get_device_rank_info()
    torch.cuda.set_device(device_info.local_rank)

    logger.info("Benchmarking barrier...")
    torch.cuda.synchronize()
    benchmark_barrier(device_info, num_iterations=args.num_iterations)

    logger.info("Benchmarking DDP forward/backward...")
    torch.cuda.synchronize()
    benchmark_ddp(device_info, num_iterations=args.num_iterations)

    logger.info("Benchmarking all_reduce...")
    torch.cuda.synchronize()
    benchmark_all_reduce(device_info, num_iterations=args.num_iterations)

    if is_distributed():
        cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Distributed/NCCL operations"
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=10,
        help="Number of iterations for each benchmark",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        help="Log level.",
    )
    args = parser.parse_args()

    update_logger_level("oumi", level=args.log_level)

    main(args)
