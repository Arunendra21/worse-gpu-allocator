# Software Requirements Specification

## 1. Purpose

The product is a deterministic, dependency-free simulator for demonstrating
bad GPU memory allocation behavior. It must be useful on machines without
CUDA, while making its intentionally poor decisions visible and testable.

## 2. Scope

The product includes a Python library, a command-line demonstration, package
metadata, automated tests, and GitHub Actions CI. It does not allocate real
GPU memory, access a network, execute user-provided code, or claim to manage
physical CUDA memory.

## 3. Functional requirements

| ID | Requirement | Acceptance evidence |
| --- | --- | --- |
| FR-01 | Accept positive integer allocation sizes. | `test_non_integer_requests_are_rejected`, unit tests |
| FR-02 | Reject invalid types, non-positive sizes, and sizes over the configured request limit. | `test_security.py` |
| FR-03 | Select GPU or CPU according to configured probability. | `test_probability_zero_always_uses_cpu`, `test_probability_one_uses_gpu_when_capacity_allows` |
| FR-04 | Fall back to CPU on GPU exhaustion by default. | `test_seeded_allocator_is_reproducible`, demo output |
| FR-05 | Provide strict GPU exhaustion behavior when configured. | `test_strict_mode_reports_gpu_exhaustion` |
| FR-06 | Release valid handles and reject double, forged, or cross-allocator frees. | `test_double_free_is_rejected_without_counter_corruption`, security tests |
| FR-07 | Expose capacity, usage, free, forgotten, CPU, and fragmentation metrics. | Snapshot assertions and CLI JSON test |
| FR-08 | Provide a machine-readable CLI result. | `test_json_cli_output_is_machine_readable` |
| FR-09 | Produce repeatable behavior for a supplied seed. | `test_seeded_allocator_is_reproducible` |
| FR-10 | Retain an optional bounded trace and allow it to be cleared safely. | `test_trace_is_bounded_and_defensively_copied`, `test_trace_can_be_cleared` |
| FR-11 | Compare the intentionally bad allocator with an honest reference model, including total resident memory. | `test_benchmark_report_is_json_serializable`, benchmark CLI |
| FR-12 | Expose an accounting invariant check for integration tests and teaching tools. | Robustness property tests |

## 4. Non-functional requirements

| ID | Requirement | Acceptance evidence |
| --- | --- | --- |
| NFR-01 | Run with Python 3.10 through 3.13 and no runtime dependencies. | GitHub Actions CI |
| NFR-02 | Be safe for concurrent library calls. | `test_concurrent_allocate_and_free_keeps_invariants` |
| NFR-03 | Avoid dynamic execution, shell invocation, network access, and secret handling. | Static security scan and `SECURITY_TEST_REPORT.md` |
| NFR-04 | Document the simulator boundary clearly. | README and this SRS |
| NFR-05 | Keep the public API small and backward compatible within the 0.x release line. | README API example and changelog |
| NFR-06 | Keep trace retention bounded by explicit configuration. | `trace_limit` validation and trace tests |
| NFR-07 | Reject invalid CLI values before starting a workload. | CLI validation tests |
| NFR-08 | Bound trace retention with a hard maximum and preserve balanced accounting under randomized workloads. | Fuzz and invariant tests |
| NFR-09 | Keep the module version synchronized with package release metadata. | Version regression test |
| NFR-10 | Bound CLI and programmatic benchmark workloads to prevent runaway execution. | Workload cap tests |
| NFR-11 | Keep allocator configuration and accounting read-only after construction. | Configuration mutation tests |
| NFR-12 | Bound CLI delay values to prevent accidental long-running processes. | Delay cap tests |
| NFR-13 | Enforce linting, strict typing, package installation, and installed CLI checks in CI. | GitHub Actions quality and matrix jobs |

## 5. Out of scope

Real CUDA bindings, performance optimization, production tensor storage,
multi-process shared memory, authentication, and remote telemetry are out of
scope for this release.
