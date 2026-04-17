# Product Changelog

## v3.2.0 — April 2026

### New Features
- **Real-time collaboration**: Multiple users can now edit documents simultaneously with live cursor tracking
- **AI summarization**: One-click document summaries powered by GPT-4o
- **Webhook v2**: New webhook format with retry logic and delivery guarantees
- **Dark mode**: Full dark mode support across all pages

### Bug Fixes
- Fixed error code 0x80004005 causing upload failures for files over 100MB
- Resolved invoice INV-2024-0392 billing discrepancy affecting Enterprise tier
- Fixed search returning stale results after bulk document updates

### Breaking Changes
- API v1 endpoints deprecated — migrate to v2 before July 2026
- `POST /documents/upload` now requires `Content-Type: multipart/form-data`

---

## v3.1.0 — March 2026

### New Features
- **Batch processing**: Process up to 500 documents in a single API call
- **Custom metadata**: Tag documents with arbitrary key-value pairs for filtering
- **SSO support**: SAML 2.0 integration for enterprise customers

### Performance
- Document ingestion is 3× faster with new parallel processing pipeline
- Search latency reduced from 450ms to 120ms p99

---

## v3.0.0 — February 2026

### Major Release
- Complete API redesign — cleaner, more consistent endpoints
- Hybrid search engine: BM25 + dense retrieval fused with RRF
- Multi-tenant architecture with full data isolation
- New SDKs for Python, TypeScript, and Go
