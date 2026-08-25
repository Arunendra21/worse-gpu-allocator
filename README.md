# A Worse GPU Memory Allocator

> It allocates memory on the GPU (sometimes).

An intentionally bad, dependency-free GPU memory allocator simulator. It is a
small joke project with a useful teaching and benchmarking surface. It makes
allocator tradeoffs visible without requiring a CUDA-capable machine.

## The features nobody asked for

- 65% chance that an allocation uses the GPU.
- CPU fallback when the GPU is full, or when the allocator simply feels like it.
- Randomly oversized buckets that waste memory.
- No block splitting and no coalescing, for maximum fragmentation.
- A 20% chance that `free()` forgets to release GPU memory.
- A linear scan, because performance is a suggestion.

## Run it

```text
python demo.py
```

Use a different deterministic run with `python demo.py --seed 42 --steps 30`.
Use `python demo.py --delay 0` for a fast run.
Use `python demo.py --steps 30 --delay 0 --json` for machine-readable metrics.

Compare it with a small sane reference allocator:

```text
python benchmark.py --steps 100
python benchmark.py --steps 100 --json
```

The library can retain a bounded event trace for teaching tools and dashboards:

```python
allocator = WorseGPUAllocator(seed=7, trace_limit=100)
allocation = allocator.allocate(4096)
print(allocator.trace())
allocator.free(allocation)
allocator.validate_invariants()
```

Trace retention is capped at one million events. Set `trace_limit=0` to keep
no events.
CLI workloads are capped at 100,000 steps to prevent accidental runaway runs.
Demo delays are capped at 60 seconds per step.

## Install it

```text
python -m pip install .
worse-gpu-demo --steps 30 --seed 42
worse-gpu-benchmark --steps 100 --json
```

The package has no runtime dependencies. It does not access CUDA, PyTorch, or
any physical device memory.

## Use it as a library

```python
from worse_gpu_allocator import WorseGPUAllocator

allocator = WorseGPUAllocator(seed=7)
tensor_memory = allocator.allocate(2 * 1024 * 1024)
print(tensor_memory.device)  # "gpu" ... probably
allocator.free(tensor_memory)
```

For callers that want a hard failure instead of the default CPU fallback, use
`WorseGPUAllocator(fallback_on_oom=False)`.

## Test it

```text
python -m unittest -v
```

For contributor checks, install the development tools and run:

```text
python -m pip install ".[dev]"
ruff check .
mypy --strict worse_gpu_allocator.py benchmark.py demo.py test_worse_gpu_allocator.py test_security.py test_features.py test_robustness.py test_fuzz.py
```

This project simulates allocation; it intentionally does not allocate real
device memory or require PyTorch/CUDA.

## Product direction

The first release is aimed at workshops, allocator demos, and test fixtures.
The next release can add trace export, side-by-side comparisons with a sane
allocator, and a browser dashboard without changing the core API.

## Requirements and security

See [SRS.md](SRS.md) for the requirements and acceptance matrix. See
[SECURITY.md](SECURITY.md) and [SECURITY_TEST_REPORT.md](SECURITY_TEST_REPORT.md)
for the threat model and the latest local security checks.
