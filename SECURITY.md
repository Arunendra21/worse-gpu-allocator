# Security Policy

## Supported versions

Security fixes are applied to the latest release on the `main` branch.

## Threat model

The library is designed to process allocation sizes and allocator handles in a
local Python process. It does not open sockets, read credentials, execute
source text, spawn subprocesses, or access physical GPU memory.

The main risks are resource exhaustion through unbounded input, misuse of
handles, and state corruption during concurrent calls. The implementation
addresses these with request limits, ownership checks, strict type validation,
and a re-entrant lock around mutable state.

## Reporting a vulnerability

Please open a private GitHub security advisory when available. For issues that
cannot be shared publicly, contact the repository owner through GitHub rather
than including exploit details in a public issue.

