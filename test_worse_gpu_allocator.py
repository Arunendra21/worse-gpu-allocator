import unittest
from unittest.mock import patch

from worse_gpu_allocator import OutOfMemoryError, WorseGPUAllocator


class WorseGPUAllocatorTests(unittest.TestCase):
    def test_seeded_allocator_is_reproducible(self) -> None:
        def run() -> tuple[list[str], dict[str, int | float]]:
            allocator = WorseGPUAllocator(gpu_capacity=10_000, seed=4)
            result: list[str] = []
            for size in (100, 700, 2_000, 300, 900):
                result.append(allocator.allocate(size).device)
            return result, allocator.snapshot()

        self.assertEqual(run(), run())

    def test_probability_zero_always_uses_cpu(self) -> None:
        allocator = WorseGPUAllocator(gpu_probability=0, seed=1)
        allocation = allocator.allocate(512)
        self.assertEqual(allocation.device, "cpu")
        self.assertEqual(allocator.snapshot()["gpu_reserved"], 0)

    def test_probability_one_uses_gpu_when_capacity_allows(self) -> None:
        allocator = WorseGPUAllocator(gpu_probability=1, gpu_capacity=10_000, seed=1)
        allocation = allocator.allocate(512)
        self.assertEqual(allocation.device, "gpu")
        self.assertGreaterEqual(allocation.reserved_bytes, allocation.requested_bytes)

    def test_freed_block_is_reused_only_at_the_same_bad_size(self) -> None:
        allocator = WorseGPUAllocator(gpu_probability=1, gpu_capacity=10_000, seed=1)
        with patch.object(allocator, "_bad_rounding", return_value=1024):
            first = allocator.allocate(512)
            allocator.free(first)
            second = allocator.allocate(512)
        self.assertEqual(second.block_index, first.block_index)

    def test_free_can_leave_a_forgotten_reservation(self) -> None:
        allocator = WorseGPUAllocator(
            gpu_probability=1,
            forget_probability=1,
            gpu_capacity=10_000,
            seed=1,
        )
        allocation = allocator.allocate(512)
        self.assertFalse(allocator.free(allocation))
        self.assertGreater(allocator.snapshot()["gpu_forgotten"], 0)

    def test_strict_mode_reports_gpu_exhaustion(self) -> None:
        allocator = WorseGPUAllocator(
            gpu_probability=1,
            gpu_capacity=1,
            fallback_on_oom=False,
            seed=1,
        )
        with self.assertRaises(OutOfMemoryError):
            allocator.allocate(512)


if __name__ == "__main__":
    unittest.main()
