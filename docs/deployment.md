# Deployment

Two free services, both deployed straight from this GitHub repository:

| Part | Host | Config in this repo |
|------|------|---------------------|
| API (FastAPI) | Render, free web service | [`render.yaml`](../render.yaml) |
| Dashboard (Streamlit) | Streamlit Community Cloud | [`app/requirements.txt`](../app/requirements.txt), [`.streamlit/config.toml`](../.streamlit/config.toml) |

## How the API gets its data

The full database is built by the pipeline and is not stored in git. The API only needs a few tables,
so `python -m econocausal.serving` copies them into
[`deploy/econocausal_serving.db`](../deploy/econocausal_serving.db) (14.9 MB), which is committed.
Render runs the API on that file, so the deployed API needs neither the raw data download nor the
machine-learning libraries. A test checks that every endpoint gives the same answers on the serving
database as on the full one.

The export is deterministic: it leaves out bookkeeping timestamps, sorts every table by its key and
rounds floats to 10 decimals, so re-running the pipeline produces a byte-identical file unless a result
actually changed. (Rounding changes answers by less than 1 part in 100 million.)

After re-running the pipeline, refresh it with:

```bash
python -m econocausal.serving
```

## 1. API on Render

1. Sign in at https://render.com with your GitHub account.
2. **New → Blueprint**, choose the `econocausal` repository and click **Apply**. Render reads
   `render.yaml`: a free Python web service named `econocausal-api` that
   - installs only `requirements-api.txt` and the `econocausal` package,
   - starts `uvicorn api.main:app` on Render's port,
   - uses Python 3.12 and the serving database,
   - checks `/health`, and
   - deploys new commits only after the CI checks pass.
3. Wait for the first build (a few minutes), then open `https://<service-name>.onrender.com/health`
   and `/docs`.

On the free plan the service sleeps after a period without traffic; the first request afterwards takes
up to about a minute while it wakes up.

## 2. Dashboard on Streamlit Community Cloud

1. Sign in at https://share.streamlit.io with your GitHub account.
2. **Create app → Deploy a public app from GitHub**:
   - Repository: `AmanDadsena/econocausal`, branch `main`
   - Main file path: `app/streamlit_app.py`
3. **Advanced settings**:
   - Python version: 3.12
   - Secrets (replace the address with your Render URL):
     ```toml
     ECONOCAUSAL_API = "https://<service-name>.onrender.com"
     ```
     Top-level secrets are also exposed as environment variables, which is how the dashboard reads
     the API address.
4. **Deploy**. Streamlit installs `app/requirements.txt` (found next to the entry file before the
   repository's full `requirements.txt`).

The dashboard retries for up to 90 seconds while the API wakes up and shows a loading message in the
meantime.

## Checklist after deploying

- [ ] `https://<api>/health` lists six tables
- [ ] `https://<api>/docs` opens, and `POST /optimal-price` with `{"stock_code": "22386"}` returns a price
- [ ] All five dashboard pages load: Overview, Elasticity, Products, Price simulator, Optimizer
- [ ] Live links added to the README
