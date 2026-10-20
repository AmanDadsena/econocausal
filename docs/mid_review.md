# Mid Review Checklist

Review window: **20 – 27 Oct 2026** · Scope: Week 1 + Week 2 work

## What the review checks, and where it is

| Review criterion | What to show | Where |
|------------------|--------------|-------|
| Week 1 and Week 2 tasks done | Roadmap with both weeks ticked; README status section | `docs/roadmap.md`, `README.md` |
| Frontend | Streamlit dashboard: headline numbers, OLS vs DML charts, categories, product explorer | `app/streamlit_app.py` |
| Backend / API | FastAPI with 8 endpoints and interactive docs | `api/main.py`, http://localhost:8000/docs |
| Database integration | SQLite tables built by the pipeline and read by the API | `src/econocausal/data/db.py` |
| Working modules | 72 passing tests; pipeline commands in the README | `tests/`, `pytest` |
| Individual contribution | Commit history on every day since 10 Oct | GitHub commits page |
| Understanding of the architecture | Architecture diagram | `README.md` → Architecture |
| Technical explanation (viva) | Concepts, decisions, results, limitations | `docs/viva_notes.md` |

## Before the call (15 minutes early)

1. Open a terminal in the project folder and activate the environment:
   ```bash
   .venv\Scripts\activate
   ```
2. Start the API (terminal 1):
   ```bash
   uvicorn api.main:app --port 8000
   ```
3. Start the dashboard (terminal 2):
   ```bash
   streamlit run app/streamlit_app.py
   ```
4. Open both and wait for the dashboard to finish loading (about 10 s the first time):
   - http://localhost:8501 (dashboard)
   - http://localhost:8000/docs (API docs)
5. Also open these tabs:
   - GitHub repository (commits page)
   - `README.md` on GitHub (status, architecture, results)
   - `reports/figures/08_method_comparison_synthetic.png` (backup chart)
6. Close unrelated tabs and apps, turn off notifications, check camera, microphone and screen sharing.
7. Join with the display name **Aman Singh Dadsena**.

## Demo script (about 5 minutes)

| Time | Show | Say |
|------|------|-----|
| 0:00 | README, top | "EconoCausal estimates how demand responds to price for an online gift retailer. The challenge is that prices are not random: they move with season and with who is buying, so a plain regression is biased." |
| 0:40 | Dashboard → *Synthetic data* tab | "To prove the method works I built synthetic data with a known elasticity of −1.8 and the same confounding as the real data. Every OLS method misses; only Double ML recovers it." |
| 1:30 | Dashboard → *Real data* tab and headline numbers | "On 126,672 real product-weeks DML gives −1.28: a 1% price rise lowers units by about 1.3%. Naive regression says 1.53%, overstating price sensitivity." |
| 2:10 | Dashboard → *Elasticity by category* | "Garden and candles are the most price sensitive, jewellery and Easter the least. Fewer products means wider intervals." |
| 2:40 | Dashboard → product explorer, search `jumbo bag` | "Each product's weekly price and units come from the API; its elasticity is the category-level DML estimate." |
| 3:10 | API docs → `GET /products/{stock_code}/elasticity`, try `22386` | "The dashboard reads everything from this REST API, which reads the SQLite database built by the pipeline. All queries are parameterised and read-only." |
| 3:40 | README → Architecture diagram | "Data flows from the raw Excel file through cleaning into SQLite, the models write results back, and the API serves them." |
| 4:10 | GitHub commits | "Commits every day since 10 October, 72 automated tests." |
| 4:30 | README → Approach table | "Next: product-level elasticities with a causal forest, robustness checks and a price optimizer." |

## Things the reviewer may try

| If asked to... | Do this |
|----------------|---------|
| Search for a famous product like "White Hanging Heart" | It is not in the panel: its weekly price was always £2.95, so its elasticity cannot be estimated. This is the filter working as designed. |
| Show the tests running | `pytest -q` (72 tests, under 2 minutes) |
| Show the code for DML | `src/econocausal/dml.py`, functions `residualize` and `final_stage` |
| Rebuild the data | `python -m econocausal.data.build` (about 15 s once the parquet file exists) |

## Backup plan

| Problem | Fallback |
|---------|----------|
| Dashboard shows "Cannot reach the API" | The API terminal is closed or crashed. Restart `uvicorn api.main:app --port 8000`, then refresh the dashboard. |
| A port is already in use | Start the API with `--port 8001`. Before starting the dashboard, point it there: `$env:ECONOCAUSAL_API="http://localhost:8001"` (PowerShell) or `set ECONOCAUSAL_API=http://localhost:8001` (Command Prompt). |
| Live demo fails completely | Show the README results tables, `reports/figures/08_method_comparison_synthetic.png` and the notebooks on GitHub (outputs are saved). |
| Internet drops | Everything runs locally. Rejoin from your phone's mobile data if needed. |

## Social media video (5 marks)

- Length about 90 seconds, screen recording with voice: problem → synthetic proof → real result → dashboard.
- Post on the platform the mentor confirms (LinkedIn by default), tag Infotact Solutions, add the GitHub link.
- Do not show the assignment letter, reference number, phone number or dashboard login.
- Submit the post link as instructed and keep a screenshot as proof.

## After the review

- Note any feedback in `docs/progress/week2.md`.
- Keep committing every day: the Final Review needs a commit on each of the 20 days before it.
