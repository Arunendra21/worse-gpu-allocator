import json
import subprocess
import sys
import unittest
from pathlib import Path

from benchmark import run_benchmark
from worse_gpu_allocator import WorseGPUAllocator

ROOT = Path(__file__).parent


class FeatureTests(unittest.TestCase):
    def test_trace_is_bounded_and_defensively_copied(self) -> None:
        allocator = WorseGPUAllocator(
            gpu_probability=0,
            trace_limit=2,
            seed=1,
        )
        first = allocator.allocate(128)
        allocator.allocate(256)
        allocator.free(first)

        events = allocator.trace()
        self.assertEqual(len(events), 2)
        self.assertEqual([event["sequence"] for event in events], [2, 3])
        events[0]["operation"] = "changed locally"
        self.assertNotEqual(allocator.trace()[0]["operation"], "changed locally")

    def test_trace_can_be_cleared(self) -> None:
        allocator = WorseGPUAllocator(gpu_probability=0, trace_limit=10, seed=1)
        allocator.allocate(128)
        self.assertEqual(allocator.snapshot()["trace_events"], 1)
        allocator.clear_trace()
        self.assertEqual(allocator.trace(), ())
        self.assertEqual(allocator.snapshot()["trace_events"], 0)

    def test_reclaim_forgotten_returns_leaked_blocks(self) -> None:
        allocator = WorseGPUAllocator(
            gpu_probability=1.0,
            forget_probability=1.0,
            trace_limit=10,
            seed=1,
        )
        handle = allocator.allocate(1000)
        self.assertFalse(allocator.free(handle))
        self.assertGreater(allocator.snapshot()["gpu_forgotten"], 0)

        reclaimed = allocator.reclaim_forgotten()
        self.assertEqual(reclaimed, 1)
        self.assertEqual(allocator.snapshot()["gpu_forgotten"], 0)
        self.assertEqual(allocator.reclaim_forgotten(), 0)
        allocator.validate_invariants()

    def test_benchmark_report_is_json_serializable(self) -> None:
        report = run_benchmark(steps=12, seed=3, capacity=1 << 20)
        encoded = json.dumps(report)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["config"]["steps"], 12)
        self.assertIn("worse", decoded)
        self.assertIn("reference", decoded)
        self.assertIn("difference", decoded)

    def test_benchmark_cli_json_output(self) -> None:
        result = subprocess.run(
            [sys.executable, "benchmark.py", "--steps", "4", "--json"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        report = json.loads(result.stdout)
        self.assertEqual(report["config"]["steps"], 4)


if __name__ == "__main__":
    unittest.main()

