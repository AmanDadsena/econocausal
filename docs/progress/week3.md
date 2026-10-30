# Week 3 Progress Notes (24 – 30 Oct 2026)

**Focus:** pricing API, full dashboard, Docker, CI, deployment setup, release 1.0.0, hardening
**Status:** all Week 3 roadmap items done except the live deployment, which needs the Render and
Streamlit Cloud accounts (configuration and guide are ready). Version 1.0.0 released.
200 automated tests, 84% line coverage, CI green.

## Work done

| Day | Work | Output |
|-----|------|--------|
| 24 Oct | Request/response schemas with strict validation; `POST /optimal-price`, `POST /simulate`, `GET /recommendations` | `api/schemas.py`, `api/main.py`, `tests/test_api_pricing.py` |
| 25 Oct | Multi-page dashboard: Overview, Elasticity, Products, Price simulator, Optimizer; `GET /elasticity/products` | `app/` |
| 26 Oct | Slim Docker images for API and dashboard; GitHub Actions CI (lint, tests, Docker build + health check); CI badge | `Dockerfile`, `Dockerfile.dashboard`, `docker-compose.yml`, `.github/workflows/ci.yml` |
| 27 Oct | Render Blueprint (validated against Render's schema); 14.9 MB serving database; Streamlit Cloud setup; API wake-up retries; deployment guide | `render.yaml`, `src/econocausal/serving.py`, [deployment.md](../deployment.md) |
| 28 Oct | One-command pipeline; full end-to-end rebuild from raw data; fixes; version 1.0.0 (tag `v1.0.0`) | `src/econocausal/pipeline.py` |
| 29 Oct | No commit (missed day) | – |
| 30 Oct | Logging; readable API errors; price input validation; tests for download and build; coverage report | `src/econocausal/log.py`, `tests/test_download.py`, `tests/test_build.py` |

## Key results

- **Reproducibility:** a full rebuild from the raw Excel file (`python -m econocausal.pipeline`, 11 minutes)
  reproduces every committed result file byte for byte. The serving database is also byte-identical across
  re-runs.
- **Deployment-ready:** the API gives the same answers from the 14.9 MB serving database as from the full
  database (all endpoints, to 1 part in 100 million), using only the slim API requirements (no ML libraries).
- **CI:** lint, 200 tests on Python 3.12 and a Docker build-and-run check pass on every push.

## Decisions and reasons

| Decision | Reason |
|----------|--------|
| Separate slim requirements for API and dashboard | The API only serves stored results; leaving out LightGBM and EconML keeps images small and deploys fast. |
| Committed serving database instead of building on the host | The full pipeline needs the raw download and ML libraries; the API needs six tables. |
| Deterministic serving export (sorted rows, no timestamps, floats rounded to 10 decimals) | Otherwise every pipeline run changed the binary file even when no result changed. |
| Render deploys only after CI passes | A broken commit cannot reach the live API. |
| Dashboard retries for up to 90 s on connection errors and 502/503/504 | Render's free plan sleeps; the first request after a pause takes up to a minute. |
| `/simulate` only accepts prices between 20% and 500% of the current price | The constant-elasticity curve is a local approximation; far from observed prices its numbers are meaningless. |
| 500 errors return only a request ID; details go to the log | Internal details (SQL, file paths) must not reach users. |

## Problems found and fixed

- **Synthetic DML results disappeared after a clean rebuild** (they were only created by a notebook, so the
  dashboard's synthetic tab would have lost its DML rows) → the pipeline now produces them.
- **Serving database changed on every run** → three causes found one by one (timestamps, floating-point noise
  of about 1e-15, row order after re-saving results) and fixed; verified with identical hashes on a re-run.
- **A half-extracted Excel file would have been treated as a finished download** on the next run → download
  and extraction now write to a temporary file and rename it only when complete; tests fail on the old code
  and pass on the new.
- **Simulator page would have sent 51 prices** (API maximum 50) → reduced to 50.
- **Optimizer page hid the catalogue** when one product's rules were infeasible → shows the message and continues.
- **API container could not start without the data folder mounted** (non-root user could not create it) → created
  in the image.
- **Python's JSON parser accepts NaN and Infinity** → rejected explicitly in every numeric request field.
- **Missed commits on 21 Oct and 29 Oct** → work done the following day; reported honestly, no backdated commits.

## Next week (31 Oct – 10 Nov)

- Live deployment (Render + Streamlit Cloud) and live links in the README
- Architecture document, model card, evaluation report (interval coverage simulation)
- Project report and presentation
- Final Review preparation and Final Review social media video

## Open questions for mentor

- Final Review date and time.
- Whether the missed commit days (21 Oct, 29 Oct) affect the Final Review GitHub requirement.
