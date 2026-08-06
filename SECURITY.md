# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.2.x   | :white_check_mark: |
| 1.1.x   | :white_check_mark: |
| 1.0.x   | :white_check_mark: |
| 0.1.x   | :x:                |

Security fixes land on the latest `1.x` release line and `main`.

## Reporting a Vulnerability

Please report security vulnerabilities by opening a private security advisory on GitHub or contacting the repository owner.

Do not open public issues for security vulnerabilities.

## Public Gradio Space

The Hugging Face Space demo (`spaces/app.py`) is rate-limited per client
(default **10 requests / minute**, override with `SPACE_RATE_LIMIT_PER_MINUTE`).
Do not rely on the public Space for production workloads — use the FastAPI
serving layer with API keys instead.
