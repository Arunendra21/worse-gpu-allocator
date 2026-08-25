"""Compare the worse allocator with a small honest reference model."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import TypedDict

from demo import MAX_STEPS, format_bytes
from worse_gpu_allocator import Allocation, WorseGPUAllocator


@dataclass(frozen=True)
class ReferenceAllocation:
    allocation_id: int
    requested_bytes: int
    device: str


Stats = dict[str, int | float]


class BenchmarkReport(TypedDict):
    config: dict[str, int]
    workload: dict[str, int]
    worse: Stats
    reference: Stats
    difference: Stats


class ReferenceGPUAllocator:
    """A compact baseline with exact sizing and no intentional leaks."""

    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("capacity must be an integer")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.gpu_used = 0
        self.cpu_bytes = 0
        self._next_id = 1
        self._allocations: dict[int, ReferenceAllocation] = {}

    def allocate(self, requested_bytes: int) -> ReferenceAllocation:
        if isinstance(requested_bytes, bool) or not isinstance(requested_bytes, int):
            raise TypeError("requested_bytes must be an integer")
        if requested_bytes <= 0:
            raise ValueError("requested_bytes must be positive")

        if self.gpu_used + requested_bytes <= self.capacity:
            device = "gpu"
            self.gpu_used += requested_bytes
        else:
            device = "cpu"
            self.cpu_bytes += requested_bytes

        allocation = ReferenceAllocation(self._next_id, requested_bytes, device)
        self._next_id += 1
        self._allocations[allocation.allocation_id] = allocation
        return allocation

    def free(self, allocation: ReferenceAllocation) -> None:
        if not isinstance(allocation, ReferenceAllocation):
            raise TypeError("allocation must be a ReferenceAllocation handle")
        current = self._allocations.get(allocation.allocation_id)
        if current is not allocation:
            raise KeyError(f"unknown allocation {allocation.allocation_id}")
        self._allocations.pop(allocation.allocation_id)
        if current.device == "gpu":
            self.gpu_used -= current.requested_bytes
        else:
            self.cpu_bytes -= current.requested_bytes

    def snapshot(self) -> Stats:
        return {
            "gpu_capacity": self.capacity,
            "gpu_reserved": self.gpu_used,
            "gpu_unreserved": self.capacity - self.gpu_used,
            "gpu_used": self.gpu_used,
            "gpu_free": 0,
            "gpu_forgotten": 0,
            "gpu_blocks": 1 if self.gpu_used else 0,
            "cpu_bytes": self.cpu_bytes,
            "active_allocations": len(self._allocations),
            "trace_events": 0,
            "fragmentation": 0.0,
        }


def run_benchmark(
    steps: int = 100,
    seed: int = 7,
    capacity: int = 32 * 1024 * 1024,
) -> BenchmarkReport:
    """Run the same workload through both allocators and return a report."""

    if isinstance(steps, bool) or not isinstance(steps, int):
        raise TypeError("steps must be an integer")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if steps > MAX_STEPS:
        raise ValueError(f"steps must be no greater than {MAX_STEPS}")

    worse = WorseGPUAllocator(
        gpu_capacity=capacity,
        gpu_probability=0.65,
        forget_probability=0.20,
        seed=seed,
    )
    reference = ReferenceGPUAllocator(capacity)
    worse_live: list[Allocation] = []
    reference_live: list[ReferenceAllocation] = []
    requested_total = 0

    for step in range(1, steps + 1):
        requested = (step * 7919) % (2 * 1024 * 1024) + 4096
        requested_total += requested
        worse_live.append(worse.allocate(requested))
        reference_live.append(reference.allocate(requested))

        if len(worse_live) > 3 and step % 3 == 0:
            worse.free(worse_live.pop(0))
            reference.free(reference_live.pop(0))

    worse_stats = worse.snapshot()
    reference_stats = reference.snapshot()
    worse_total = worse_stats["gpu_reserved"] + worse_stats["cpu_bytes"]
    reference_total = reference_stats["gpu_reserved"] + reference_stats["cpu_bytes"]
    return {
        "config": {"steps": steps, "seed": seed, "capacity": capacity},
        "workload": {"requested_bytes": requested_total},
        "worse": worse_stats,
        "reference": reference_stats,
        "difference": {
            "gpu_reserved_bytes": worse_stats["gpu_reserved"] - reference_stats["gpu_reserved"],
            "cpu_bytes": worse_stats["cpu_bytes"] - reference_stats["cpu_bytes"],
            "total_resident_bytes": worse_total - reference_total,
            "fragmentation": worse_stats["fragmentation"] - reference_stats["fragmentation"],
        },
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    if parsed > MAX_STEPS:
        raise argparse.ArgumentTypeError(
            f"must be no greater than {MAX_STEPS}"
        )
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=_positive_int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--capacity", type=_positive_int, default=32 * 1024 * 1024)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run_benchmark(args.steps, args.seed, args.capacity)

    if args.json:
        print(json.dumps(report, sort_keys=True))
        return

    worse = report["worse"]
    reference = report["reference"]
    difference = report["difference"]
    print("A Worse GPU Memory Allocator: benchmark")
    print(f"Workload: {report['config']['steps']} steps, seed {report['config']['seed']}")
    print()
    print(" metric                 worse          reference       difference")
    print(" --------------------  -------------  -------------  -------------")
    print(
        f" GPU reserved          {format_bytes(worse['gpu_reserved']):>13}  "
        f"{format_bytes(reference['gpu_reserved']):>13}  "
        f"{format_bytes(difference['gpu_reserved_bytes']):>13}"
    )
    print(
        f" CPU fallback          {format_bytes(worse['cpu_bytes']):>13}  "
        f"{format_bytes(reference['cpu_bytes']):>13}  "
        f"{format_bytes(difference['cpu_bytes']):>13}"
    )
    print(
        f" Total resident        {format_bytes(worse['gpu_reserved'] + worse['cpu_bytes']):>13}  "
        f"{format_bytes(reference['gpu_reserved'] + reference['cpu_bytes']):>13}  "
        f"{format_bytes(difference['total_resident_bytes']):>13}"
    )
    print(
        f" Fragmentation         {worse['fragmentation']:>12.1%}  "
        f"{reference['fragmentation']:>12.1%}  "
        f"{difference['fragmentation']:>12.1%}"
    )


if __name__ == "__main__":
    main()
