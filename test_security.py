import json
import subprocess
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from worse_gpu_allocator import Allocation, WorseGPUAllocator

ROOT = Path(__file__).parent


class SecurityAndRobustnessTests(unittest.TestCase):
    def test_non_integer_requests_are_rejected(self) -> None:
        allocator = WorseGPUAllocator(seed=1)
        for value in (1.5, "1024", None, True):
            with self.subTest(value=value), self.assertRaises(TypeError):
                allocator.allocate(value)  # type: ignore[arg-type]

    def test_request_limit_prevents_resource_abuse(self) -> None:
        allocator = WorseGPUAllocator(max_request_bytes=1024, seed=1)
        with self.assertRaises(ValueError):
            allocator.allocate(1025)
        self.assertEqual(allocator.snapshot()["active_allocations"], 0)

    def test_cross_allocator_handle_cannot_be_freed(self) -> None:
        first = WorseGPUAllocator(gpu_probability=0, seed=1)
        second = WorseGPUAllocator(gpu_probability=0, seed=1)
        handle = first.allocate(512)
        with self.assertRaises(KeyError):
            second.free(handle)
        self.assertEqual(first.snapshot()["active_allocations"], 1)

    def test_forged_handle_cannot_be_freed(self) -> None:
        allocator = WorseGPUAllocator(gpu_probability=0, seed=1)
        handle = allocator.allocate(512)
        forged = replace(handle)
        self.assertIsInstance(forged, Allocation)
        with self.assertRaises(KeyError):
            allocator.free(forged)
        self.assertEqual(allocator.snapshot()["active_allocations"], 1)

    def test_double_free_is_rejected_without_counter_corruption(self) -> None:
        allocator = WorseGPUAllocator(gpu_probability=0, seed=1)
        handle = allocator.allocate(512)
        self.assertTrue(allocator.free(handle))
        with self.assertRaises(KeyError):
            allocator.free(handle)
        self.assertEqual(allocator.snapshot()["cpu_bytes"], 0)

    def test_concurrent_allocate_and_free_keeps_invariants(self) -> None:
        allocator = WorseGPUAllocator(
            gpu_capacity=1 << 20,
            gpu_probability=0.5,
            forget_probability=0,
            seed=4,
        )

        def allocate_and_free(_: int) -> None:
            handle = allocator.allocate(256)
            allocator.free(handle)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(allocate_and_free, range(100)))

        stats = allocator.snapshot()
        self.assertEqual(stats["active_allocations"], 0)
        self.assertEqual(stats["cpu_bytes"], 0)
        self.assertEqual(stats["gpu_used"], 0)
        self.assertGreaterEqual(stats["gpu_reserved"], stats["gpu_free"])

    def test_json_cli_output_is_machine_readable(self) -> None:
        result = subprocess.run(
            [sys.executable, "demo.py", "--steps", "2", "--delay", "0", "--json"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        self.assertIn("gpu_capacity", payload)
        self.assertIn("active_allocations", payload)


if __name__ == "__main__":
    unittest.main()

