# Changelog

## [Unreleased]
### Added
- Multi-model meteogram generation with icing diagnostics and merged JSON output.
- Plotly-based frontend (`web/`) and FastAPI static/API server.
- Environment-driven configuration, Makefile/scripts, and CI/test scaffolding.
- Systemd sample units for scheduled fetch + serving.

### Fixed
- Longitude normalization for 0–360° grids and missing-run auto fallback.

### TODO
- Docker image + compose recipe.
- Retry/backoff + cache-last-good improvements.
