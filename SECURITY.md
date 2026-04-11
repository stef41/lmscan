# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.3.x   | :white_check_mark: |
| < 0.3   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in lmscan, please report it responsibly:

1. **Do NOT open a public issue.**
2. Email **security@lmscan.dev** or use [GitHub Security Advisories](https://github.com/stef41/lmscan/security/advisories/new).
3. Include steps to reproduce and impact assessment.

We will acknowledge receipt within 48 hours and provide a fix timeline within 7 days.

## Scope

lmscan processes text locally. It does not:
- Send data to external servers
- Require API keys or credentials
- Execute user-provided code

Security concerns are primarily around dependency supply chain and input handling.
