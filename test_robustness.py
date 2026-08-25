import random
import re
import subprocess
import sys
import unittest
from pathlib import Path

from benchmark import ReferenceGPUAllocator
from demo import MAX_DELAY_SECONDS, MAX_STEPS
from worse_gpu_allocator import (
    MAX_TRACE_LIMIT,
    Allocation,
    WorseGPUAllocator,
    __version__,
)

ROOT = Path(__file__).parent


class RobustnessTests(unittest.TestCase):
    def test_module_version_matches_current_release(self) -> None:
        self.assertEqual(__version__, "0.9.1")
        metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'^version = "([^"]+)"$', metadata, re.MULTILINE)
        if match is None:
            self.fail("pyproject.toml has no project version")
        self.assertEqual(match.group(1), __version__)

    def test_trace_limit_has_a_hard_cap(self) -> None:
        with self.assertRaises(ValueError):
            WorseGPUAllocator(trace_limit=MAX_TRACE_LIMIT + 1)

    def test_configuration_and_accounting_are_read_only(self) -> None:
        allocator = WorseGPUAllocator(seed=1)
        for name, value in (
            ("gpu_capacity", 1),
            ("gpu_probability", 0),
            ("forget_probability", 0),
            ("fallback_on_oom", False),
            ("max_request_bytes", 1),
            ("trace_limit", 1),
            ("cpu_bytes", 1),
        ):
            with self.subTest(name=name), self.assertRaises(AttributeError):
                setattr(allocator, name, value)
        allocator.validate_invariants()

    def test_randomized_workloads_preserve_invariants(self) -> None:
        for seed in range(20):
            with self.subTest(seed=seed):
                chooser = random.Random(seed)
                allocator = WorseGPUAllocator(
                    gpu_capacity=1 << 20,
                    gpu_probability=0.55,
                    forget_probability=0,
                    trace_limit=32,
                    seed=seed,
                )
                live: list[Allocation] = []
                for _ in range(200):
                    if live and chooser.random() < 0.45:
                        allocator.free(live.pop(chooser.randrange(len(live))))
                    else:
                        live.append(allocator.allocate(chooser.randint(1, 32_768)))
                    allocator.validate_invariants()

                for allocation in live:
                    allocator.free(allocation)
                allocator.validate_invariants()
                stats = allocator.snapshot()
                self.assertEqual(stats["active_allocations"], 0)
                self.assertEqual(stats["cpu_bytes"], 0)
                self.assertEqual(stats["gpu_used"], 0)
                self.assertGreaterEqual(stats["gpu_unreserved"], 0)

    def test_strict_oom_is_atomic(self) -> None:
        allocator = WorseGPUAllocator(
            gpu_capacity=1,
            fallback_on_oom=False,
            gpu_probability=1,
            seed=1,
        )
        before = allocator.snapshot()
        with self.assertRaises(MemoryError):
            allocator.allocate(1024)
        self.assertEqual(allocator.snapshot(), before)
        allocator.validate_invariants()

    def test_reference_cross_free_does_not_corrupt_state(self) -> None:
        first = ReferenceGPUAllocator(1024)
        second = ReferenceGPUAllocator(1024)
        handle = first.allocate(128)
        with self.assertRaises(KeyError):
            second.free(handle)
        self.assertEqual(first.snapshot()["active_allocations"], 1)
        first.free(handle)
        self.assertEqual(first.snapshot()["active_allocations"], 0)

    def test_cli_rejects_invalid_workload_sizes(self) -> None:
        for script in ("demo.py", "benchmark.py"):
            with self.subTest(script=script):
                result = subprocess.run(
                    [sys.executable, script, "--steps", "0"],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("must be positive", result.stderr)

    def test_cli_rejects_unbounded_workloads(self) -> None:
        for script in ("demo.py", "benchmark.py"):
            with self.subTest(script=script):
                result = subprocess.run(
                    [sys.executable, script, "--steps", str(MAX_STEPS + 1)],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("no greater", result.stderr)

    def test_cli_rejects_non_finite_delay(self) -> None:
        for value in ("nan", "inf", "-inf"):
            with self.subTest(value=value):
                result = subprocess.run(
                    [sys.executable, "demo.py", "--steps", "1", f"--delay={value}"],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("must be finite", result.stderr)

    def test_cli_rejects_runaway_delay(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "demo.py",
                "--steps",
                "1",
                f"--delay={MAX_DELAY_SECONDS + 1}",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no greater", result.stderr)


if __name__ == "__main__":
    unittest.main()
