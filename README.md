# Musanze Harvest Intelligence Service

> The dataset used in this project is **synthetic**.

| Field | Value |
|---|---|
| Full name | `<YOUR FULL NAME>` |
| Registration number | 25/27916 |
| Institutional username | ug2527916@ines.ac.rw |
| Verification code | AI-A2-7916 |
| Repository URL / final commit hash | `<fill in>` / `<fill in>` |

## Problem
A Musanze potato cooperative needs to (1) estimate harvest tons, (2) flag high-risk farms
(yield per hectare below the dataset median) and (3) group similar farms. Training and CLI prediction run offline.

## Setup (Python 3.10+; write the version/OS you tested)
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py --student-id ug2527916@ines.ac.rw --seed 7916
python predict.py --json samples/valid.json
python predict.py --json samples/invalid.json     # exits with code 2 and field-level errors
pytest -q
uvicorn app:app --host 127.0.0.1 --port 8000      # docs at http://127.0.0.1:8000/docs
```

## API examples
```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/model-info
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @samples/valid.json
```
Invalid input returns HTTP 422 with a per-field error list; missing models return 503.

## Layout
`run_all.py` (pipeline) · `predict.py` (CLI) · `app.py` (FastAPI) · `data/generate_data.py` ·
`src/` (config, data_generation, validation, preprocessing, modeling, clustering, reporting, schemas, inference) ·
`tests/` · `artifacts/` · `models/` · `samples/` · `docs/` · `evidence/`

## Design decisions
- Seed `LLLL` (last 4 registration digits) drives data, split, CV folds and all model random states.
- Duplicates are removed after validation; missing values are median-imputed **inside the Pipeline** (train-only fit).
- One stratified 80/20 split (`random_state=LLLL`); models are selected by **5-fold CV on the training set**, then reported once on the test set.
- Features exclude `yield_tons`, yield-per-hectare and `high_risk` (leakage guard in `validation.py`).
- `high_risk` is defined from the median of the full generated dataset, as the brief requires. It is a label definition, not a fitted preprocessing step.
- Because the generator injects missing values before copying duplicates, a duplicated row can carry a missing value, so counts can slightly exceed `m`.
- No extra packages beyond the brief except `uvicorn`, `pydantic` (FastAPI stack) and `httpx` (FastAPI TestClient).

## Responsible AI
Intended use: classroom decision-support prototype. Data: synthetic, so metrics say nothing about real Rwandan farms.
Limitations: simple linear signal, no soil type/pests/prices/geography. Risks: over-trust, drift on real data.
Prohibited: finance, insurance or public-service eligibility decisions. A human (agronomist/field officer) must review every high-risk flag.

## Known limitations
Clusters are weak (inputs are independent by construction); the report must say so honestly.
