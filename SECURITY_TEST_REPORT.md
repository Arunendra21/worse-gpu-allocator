# Security Test Report

## Test date

2026-08-23

## Scope

The Python library, CLI entry point, package metadata, and repository source.
No physical GPU or remote service was targeted.

## Checks performed

- Invalid type and oversized allocation inputs were rejected.
- Cross-allocator, forged, and double-free handles were rejected.
- Concurrent allocate and free operations preserved accounting invariants.
- CLI JSON output was parsed without executing content.
- Trace retention was bounded and returned through defensive copies.
- Trace limits above one million events and non-finite CLI delays were rejected.
- CLI and programmatic benchmark workloads above 100,000 steps were rejected.
- Configuration and accounting mutation attempts were rejected.
- Finite but excessive CLI delays were rejected before execution.
- Reference allocator non-handle input was rejected with a type error.
- Benchmark JSON output was parsed and its total resident memory comparison was verified.
- Randomized workloads repeatedly checked balanced CPU and GPU accounting.
- The module version was checked against the release version to prevent stale-version distribution bugs.
- Source was reviewed for `eval`, `exec`, dynamic imports, shell calls,
  sockets, HTTP clients, credential reads, and unsafe deserialization.
- The 30-test suite passed locally, including 54,000 randomized operations.
- Ruff and mypy reported no findings across product and test code.
- CI installs the package and enforces Ruff plus strict mypy on every change.
- GitHub Actions ran the test suite on Python 3.10, 3.11, 3.12, and 3.13.
- CI uses read-only repository permissions and a five-minute job timeout.

## Result

Pass for the defined local threat model. The product remains a simulator and
must not be presented as a real CUDA allocator or used as a security boundary.
