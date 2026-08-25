"""A deliberately bad GPU memory allocator.

This is a small, dependency-free simulator intended as a teaching/demo project.
It models the sort of allocator you should not ship:

* allocations go to the GPU only with a probability;
* requests are padded into awkwardly sized blocks;
* freed blocks are not coalesced, so fragmentation grows;
* some frees are "forgotten" and remain reserved;
* allocation performs a deliberately slow linear scan.

The allocator never touches a real GPU. That makes the joke reproducible on any
machine and keeps the project useful in tests and demos.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from threading import RLock
from typing import Literal

__version__ = "0.9.0"
MAX_TRACE_LIMIT = 1_000_000
__all__ = [
    "MAX_TRACE_LIMIT",
    "Allocation",
    "OutOfMemoryError",
    "TraceEvent",
    "WorseGPUAllocator",
    "__version__",
]


Device = Literal["gpu", "cpu"]
TraceEvent = dict[str, int | str | bool]


@dataclass(frozen=True)
class Allocation:
    """A handle returned by :meth:`WorseGPUAllocator.allocate`."""

    allocation_id: int
    requested_bytes: int
    reserved_bytes: int
    device: Device
    block_index: int
    _owner_token: object = field(repr=False, compare=False)


@dataclass
class Block:
    """One reserved block in the fake GPU arena."""

    size: int
    allocation_id: int | None = None
    forgotten: bool = False

    @property
    def free(self) -> bool:
        return self.allocation_id is None and not self.forgotten


class OutOfMemoryError(MemoryError):
    """Raised when the fake GPU cannot satisfy a request."""


class WorseGPUAllocator:
    """An allocator whose design goals are inconsistency and regret.

    Args:
        gpu_capacity: Simulated GPU capacity in bytes.
        gpu_probability: Chance that an allocation actually uses the GPU.
        forget_probability: Chance that ``free`` forgets to release a GPU block.
        seed: Seed for reproducible bad decisions.
        trace_limit: Maximum number of events retained, or zero to disable tracing.
    """

    def __init__(
        self,
        gpu_capacity: int = 64 * 1024 * 1024,
        gpu_probability: float = 0.65,
        forget_probability: float = 0.20,
        fallback_on_oom: bool = True,
        max_request_bytes: int = 1 << 40,
        trace_limit: int = 0,
        seed: int | None = None,
    ) -> None:
        if isinstance(gpu_capacity, bool) or not isinstance(gpu_capacity, int):
            raise TypeError("gpu_capacity must be an integer")
        if gpu_capacity <= 0:
            raise ValueError("gpu_capacity must be positive")
        self._validate_probability("gpu_probability", gpu_probability)
        self._validate_probability("forget_probability", forget_probability)
        if not isinstance(fallback_on_oom, bool):
            raise TypeError("fallback_on_oom must be a boolean")
        if isinstance(max_request_bytes, bool) or not isinstance(max_request_bytes, int):
            raise TypeError("max_request_bytes must be an integer")
        if max_request_bytes <= 0:
            raise ValueError("max_request_bytes must be positive")
        if isinstance(trace_limit, bool) or not isinstance(trace_limit, int):
            raise TypeError("trace_limit must be an integer")
        if trace_limit < 0:
            raise ValueError("trace_limit cannot be negative")
        if trace_limit > MAX_TRACE_LIMIT:
            raise ValueError(f"trace_limit cannot exceed {MAX_TRACE_LIMIT}")

        self._gpu_capacity = gpu_capacity
        self._gpu_probability = gpu_probability
        self._forget_probability = forget_probability
        self._fallback_on_oom = fallback_on_oom
        self._max_request_bytes = max_request_bytes
        self._trace_limit = trace_limit
        self._random = random.Random(seed)
        self._blocks: list[Block] = []
        self._allocations: dict[int, Allocation] = {}
        self._next_id = 1
        self._cpu_bytes = 0
        self._lock = RLock()
        self._owner_token = object()
        self._trace_events: list[TraceEvent] = []
        self._trace_sequence = 0

    @property
    def gpu_capacity(self) -> int:
        return self._gpu_capacity

    @property
    def gpu_probability(self) -> float:
        return self._gpu_probability

    @property
    def forget_probability(self) -> float:
        return self._forget_probability

    @property
    def fallback_on_oom(self) -> bool:
        return self._fallback_on_oom

    @property
    def max_request_bytes(self) -> int:
        return self._max_request_bytes

    @property
    def trace_limit(self) -> int:
        return self._trace_limit

    @property
    def cpu_bytes(self) -> int:
        return self._cpu_bytes

    def allocate(self, requested_bytes: int) -> Allocation:
        """Reserve memory, occasionally on the GPU, and return its handle."""

        if isinstance(requested_bytes, bool) or not isinstance(requested_bytes, int):
            raise TypeError("requested_bytes must be an integer")
        if requested_bytes <= 0:
            raise ValueError("requested_bytes must be positive")
        if requested_bytes > self.max_request_bytes:
            raise ValueError(
                f"requested_bytes exceeds the {self.max_request_bytes} byte limit"
            )

        with self._lock:
            if self._random.random() >= self.gpu_probability:
                return self._allocate_cpu(requested_bytes, reason="probability")

            reserved_bytes = self._bad_rounding(requested_bytes)
            block_index = self._find_first_fit(reserved_bytes)
            if block_index is None:
                if not self.fallback_on_oom:
                    self._record_event(
                        {
                            "operation": "allocate_failed",
                            "reason": "gpu_oom",
                            "requested_bytes": requested_bytes,
                            "device": "gpu",
                        }
                    )
                    raise OutOfMemoryError(
                        f"GPU request for {requested_bytes} bytes could not be reserved"
                    )
                # A real allocator might compact or retry. This one gives up and
                # quietly sends the work to CPU, which is worse in a different way.
                return self._allocate_cpu(requested_bytes, reason="gpu_oom")

            allocation = Allocation(
                allocation_id=self._next_id,
                requested_bytes=requested_bytes,
                reserved_bytes=reserved_bytes,
                device="gpu",
                block_index=block_index,
                _owner_token=self._owner_token,
            )
            self._next_id += 1
            self._blocks[block_index].allocation_id = allocation.allocation_id
            self._allocations[allocation.allocation_id] = allocation
            self._record_event(
                {
                    "operation": "allocate",
                    "allocation_id": allocation.allocation_id,
                    "requested_bytes": requested_bytes,
                    "reserved_bytes": reserved_bytes,
                    "device": "gpu",
                }
            )
            return allocation

    def free(self, allocation: Allocation) -> bool:
        """Free an allocation, unless the allocator randomly forgets.

        Returns ``True`` when the memory was actually released and ``False``
        when it became a forgotten reservation.
        """

        if not isinstance(allocation, Allocation):
            raise TypeError("allocation must be an Allocation handle")

        with self._lock:
            if allocation._owner_token is not self._owner_token:
                raise KeyError("allocation belongs to another allocator")

            current = self._allocations.get(allocation.allocation_id)
            if current is not allocation:
                raise KeyError(f"unknown allocation {allocation.allocation_id}")
            self._allocations.pop(allocation.allocation_id)

            if current.device == "cpu":
                self._cpu_bytes -= current.reserved_bytes
                self._record_event(
                    {
                        "operation": "free",
                        "allocation_id": current.allocation_id,
                        "released": True,
                        "device": "cpu",
                    }
                )
                return True

            block = self._blocks[current.block_index]
            if self._random.random() < self.forget_probability:
                block.allocation_id = None
                block.forgotten = True
                self._record_event(
                    {
                        "operation": "free",
                        "allocation_id": current.allocation_id,
                        "released": False,
                        "device": "gpu",
                        "reason": "forgotten",
                    }
                )
                return False

            block.allocation_id = None
            self._record_event(
                {
                    "operation": "free",
                    "allocation_id": current.allocation_id,
                    "released": True,
                    "device": "gpu",
                }
            )
            return True

    def snapshot(self) -> dict[str, int | float]:
        """Return simple metrics suitable for a CLI or dashboard."""

        with self._lock:
            reserved = sum(block.size for block in self._blocks)
            used = sum(
                block.size
                for block in self._blocks
                if block.allocation_id is not None
            )
            free = sum(block.size for block in self._blocks if block.free)
            forgotten = sum(block.size for block in self._blocks if block.forgotten)
            return {
                "gpu_capacity": self.gpu_capacity,
                "gpu_reserved": reserved,
                "gpu_unreserved": self.gpu_capacity - reserved,
                "gpu_used": used,
                "gpu_free": free,
                "gpu_forgotten": forgotten,
                "gpu_blocks": len(self._blocks),
                "cpu_bytes": self.cpu_bytes,
                "active_allocations": len(self._allocations),
                "trace_events": len(self._trace_events),
                "fragmentation": round(1 - (free / reserved), 3) if reserved else 0.0,
            }

    def validate_invariants(self) -> None:
        """Raise ``RuntimeError`` if internal accounting has become invalid.

        This is intentionally public so tests and teaching tools can verify
        allocator state after a workload without relying on private fields.
        """

        with self._lock:
            reserved = sum(block.size for block in self._blocks)
            if reserved > self.gpu_capacity:
                raise RuntimeError("GPU reservations exceed capacity")
            if self.cpu_bytes < 0:
                raise RuntimeError("CPU accounting is negative")
            if len(self._trace_events) > self.trace_limit:
                raise RuntimeError("trace exceeds configured limit")

            cpu_total = sum(
                allocation.reserved_bytes
                for allocation in self._allocations.values()
                if allocation.device == "cpu"
            )
            if cpu_total != self.cpu_bytes:
                raise RuntimeError("CPU allocation and byte indexes disagree")

            used = 0
            free = 0
            forgotten = 0
            for block in self._blocks:
                if block.size <= 0:
                    raise RuntimeError("GPU block size is not positive")
                if block.forgotten and block.allocation_id is not None:
                    raise RuntimeError("forgotten block still has an allocation")
                if block.forgotten:
                    forgotten += block.size
                elif block.free:
                    free += block.size
                else:
                    used += block.size
            if used + free + forgotten != reserved:
                raise RuntimeError("GPU block accounting does not balance")

            block_ids = {
                block.allocation_id
                for block in self._blocks
                if block.allocation_id is not None
            }
            gpu_ids = {
                allocation.allocation_id
                for allocation in self._allocations.values()
                if allocation.device == "gpu"
            }
            if block_ids != gpu_ids:
                raise RuntimeError("GPU block and allocation indexes disagree")

            for allocation in self._allocations.values():
                if allocation.device == "cpu":
                    if allocation.block_index != -1:
                        raise RuntimeError("CPU allocation has a GPU block index")
                    continue
                if not 0 <= allocation.block_index < len(self._blocks):
                    raise RuntimeError("GPU allocation has an invalid block index")
                if self._blocks[allocation.block_index].allocation_id != allocation.allocation_id:
                    raise RuntimeError("GPU allocation points to the wrong block")

    def trace(self) -> tuple[TraceEvent, ...]:
        """Return a copy of the bounded event trace."""

        with self._lock:
            return tuple(dict(event) for event in self._trace_events)

    def clear_trace(self) -> None:
        """Discard all retained trace events."""

        with self._lock:
            self._trace_events.clear()

    def _allocate_cpu(self, requested_bytes: int, reason: str) -> Allocation:
        allocation = Allocation(
            allocation_id=self._next_id,
            requested_bytes=requested_bytes,
            reserved_bytes=requested_bytes,
            device="cpu",
            block_index=-1,
            _owner_token=self._owner_token,
        )
        self._next_id += 1
        self._allocations[allocation.allocation_id] = allocation
        self._cpu_bytes += requested_bytes
        self._record_event(
            {
                "operation": "allocate",
                "allocation_id": allocation.allocation_id,
                "requested_bytes": requested_bytes,
                "reserved_bytes": requested_bytes,
                "device": "cpu",
                "reason": reason,
            }
        )
        return allocation

    def _record_event(self, event: TraceEvent) -> None:
        if self.trace_limit == 0:
            return
        self._trace_sequence += 1
        event = dict(event)
        event["sequence"] = self._trace_sequence
        if len(self._trace_events) >= self.trace_limit:
            self._trace_events.pop(0)
        self._trace_events.append(event)

    @staticmethod
    def _validate_probability(name: str, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number")
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")

    def _find_first_fit(self, size: int) -> int | None:
        # Intentional O(n) scan. It also refuses to split a larger block,
        # creating needless waste and more fragmentation.
        for index, block in enumerate(self._blocks):
            if block.free and block.size == size:
                return index

        reserved = sum(block.size for block in self._blocks)
        if reserved + size > self.gpu_capacity:
            return None

        self._blocks.append(Block(size=size))
        return len(self._blocks) - 1

    def _bad_rounding(self, size: int) -> int:
        """Round to a random bucket, occasionally wasting almost 2x memory."""

        bucket = self._random.choice((256, 1024, 4096, 16384))
        rounded = ((size + bucket - 1) // bucket) * bucket
        if self._random.random() < 0.12:
            rounded *= 2
        return rounded
