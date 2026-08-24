# Security Policy

## Reporting a vulnerability

Please **do not open a public issue** for security vulnerabilities.

Instead, report privately to the maintainers via email or a private channel
(reach out through the repository's GitHub discussions/contact options).
You'll get an acknowledgment within 48 hours and a fix plan as soon as
possible.

## What we care about

- Credential handling: this project passes API keys to upstreams. Never log
  keys, never commit them, never put them in dashboard/metrics output.
- The gateway's auth layer (`FUSION_GATEWAY_API_KEY`): verify it works as
  documented (only `/v1/*` is protected; health/metrics/dashboard stay open).
- Remote code execution surfaces: the gateway only makes HTTP calls to the
  configured upstream; do not add endpoints that eval/exec user input.

## Supported versions

Only the latest `main` is supported. There are no release branches yet.

## Disclosure

We prefer coordinated disclosure. Please give us a reasonable window
(14 days by default) to fix before publishing details.
