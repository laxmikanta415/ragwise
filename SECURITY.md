# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |

## Reporting a vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Email **lax@insummary.com** with:
- Subject: `[SECURITY] ragwise`
- A description of the vulnerability
- Steps to reproduce
- Potential impact

We aim to acknowledge reports within **48 hours** and provide a fix timeline within **7 days**.

## Scope

Areas of particular concern:

- Prompt injection via ingested documents reaching the LLM
- Path traversal in `ingest()` when accepting user-supplied paths
- SQL injection in `PgVectorStore` query parameters
- Unsafe deserialization in store backends

## Out of scope

- Issues in optional dependencies (report upstream to the relevant project)
- Vulnerabilities requiring physical access to the host machine
