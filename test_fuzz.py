import random
import unittest

from benchmark import ReferenceGPUAllocator, run_benchmark
from demo import MAX_STEPS
from worse_gpu_allocator import Allocation, WorseGPUAllocator


class FuzzWorkloadTests(unittest.TestCase):
    def test_programmatic_benchmark_rejects_unbounded_workloads(self) -> None:
        with self.assertRaises(ValueError):
            run_benchmark(steps=MAX_STEPS + 1)

    def test_reference_rejects_non_handles(self) -> None:
        allocator = ReferenceGPUAllocator(1024)
        with self.assertRaises(TypeError):
            allocator.free(object())  # type: ignore[arg-type]

    def test_many_seeded_adversarial_workloads(self) -> None:
        for seed in range(100):
            with self.subTest(seed=seed):
                chooser = random.Random(seed)
                allocator = WorseGPUAllocator(
                    gpu_capacity=2 * 1024 * 1024,
                    gpu_probability=chooser.random(),
                    forget_probability=chooser.random() * 0.4,
                    trace_limit=64,
                    seed=seed,
                )
                live: list[Allocation] = []
                for _ in range(500):
                    if live and chooser.random() < 0.5:
                        handle = live.pop(chooser.randrange(len(live)))
                        allocator.free(handle)
                    else:
                        size = chooser.choice(
                            (1, 2, 255, 256, 257, 1024, 4095, 4096, 65_536)
                        )
                        live.append(allocator.allocate(size))
                    allocator.validate_invariants()

                for handle in live:
                    allocator.free(handle)
                allocator.validate_invariants()

                stats = allocator.snapshot()
                self.assertGreaterEqual(stats["gpu_unreserved"], 0)
                self.assertLessEqual(stats["gpu_reserved"], stats["gpu_capacity"])
                self.assertEqual(stats["active_allocations"], 0)


if __name__ == "__main__":
    unittest.main()
