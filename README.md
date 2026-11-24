# Malware Analysis & Evaluation Agent

MSAIE Project 4 (Quantic): an ML model that classifies Windows PE-file feature records as **goodware (0)** or **malware (1)**, plus a deployed AI agent (Anthropic Claude) that calls that model through tools for single-record prediction, batch prediction and labeled-data evaluation.

* **Live app:** see [`deployed.md`](deployed.md)
* **ML and system design, CV table, hold-out results:** [`evaluation-and-design.md`](evaluation-and-design.md)
* **Live agent evaluation (actual AI model):** [`agent-evaluation.md`](agent-evaluation.md)
* **AI coding tools used:** [`ai-tooling.md`](ai-tooling.md)

**Results at a glance.** The production model is LightGBM v1.0.0 (threshold 0.5). It scored 10-fold CV AUC 0.9988 ± 0.0004 and accuracy 0.9883 ± 0.0017. On the held-out 20% test set it scored **AUC 0.9989** and **accuracy 98.93%** (TN 4178, FP 41, FN 52, TP 4406).

## Repository layout

```
malware_ml/          schema, preprocessing, PyTorch MLP wrapper, model registry, production inference
train.py             full model development: split, ablation, tuning, 10-fold CV, selection, refit, save
eval.py              one-time hold-out evaluation of the frozen artifact
artifacts/           model.joblib (preprocessing + LightGBM) and metadata.json (schema, labels, threshold, version)
reports/             split info, training-data profile, feature ablation, tuning grid, CV results, hold-out results, training log
app/                 Flask app: agent loop (agent.py), tools (tools.py), session store, chat UI
samples/             small CSV/JSON files for demos and tests (no executables)
scripts/             dataset download, live agent-evaluation runner
tests/               unit, integration and mocked-AI agent tests
.github/workflows/   CI/CD: tests → deploy (Render hook, exact commit) → /health smoke test
render.yaml          Render service definition (auto-deploy off)
```

## 1. Environment setup

Python 3.13.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
# App + tests only:
pip install -r requirements.txt
# Full model development (adds XGBoost, CatBoost, CPU PyTorch):
pip install -r requirements-train.txt --extra-index-url https://download.pytorch.org/whl/cpu
cp .env.example .env                 # then fill in ANTHROPIC_API_KEY; never commit .env
```

## 2. Reproduce training and evaluation

```bash
python scripts/download_data.py      # fetches the dataset at the pinned commit and verifies SHA-256
python train.py                      # ~1 h on 2 CPU cores; writes artifacts/ and reports/
python eval.py                       # hold-out AUC, accuracy, confusion matrix -> reports/holdout_results.json
```

* `python train.py --quick` runs a smoke version (4,000 rows, 3 folds, 1 setting per model) in about 2 minutes.
* Seeds are fixed at 42 for the split, the folds and every model.
* The dataset version, feature schema, label mapping, threshold, model version and library versions are recorded in `malware_ml/schema.py` and `artifacts/metadata.json`.
* `eval.py --csv other_labeled.csv` evaluates the artifact on any labeled file.

## 3. Run the app locally

```bash
set -a; source .env; set +a          # load ANTHROPIC_API_KEY etc. into the environment
gunicorn wsgi:app --workers 1 --threads 4 --bind 0.0.0.0:5000
# or: python wsgi.py
```

Open http://localhost:5000.

* The model is loaded once at startup from `artifacts/` by `malware_ml.inference.ProductionModel.load()`. Set `MODEL_DIR` to load from a different folder.
* If the artifact is missing or broken, startup fails rather than serving without a model.
* Keep **one** worker: session data lives in process memory.

## 4. Tests

```bash
python -m pytest -v tests
```

* **Unit tests** (`tests/test_ml.py`): schema validation, preprocessing, de-duplication, model wrapper and threshold, class mapping, and metrics checked against hand-computed values.
* **Integration tests** (`tests/test_app.py`): `/health`, `/api/predict`, batch and download, labeled evaluation, missing labels, single class, upload validation, session isolation.
* **Agent tests** (`tests/test_agent.py`): a scripted mock replaces Claude, so tool dispatch, argument validation, conditional prediction (permitted, withheld, and failed evaluation), false-negative follow-up, tool failure, LLM timeout and the tool-call limit are all checked deterministically. No API key is needed.

## 5. CI/CD and deployment

`.github/workflows/ci-cd.yml`:
1. **test** runs on every pull request and on every push to `main`.
2. **deploy** runs only on `main`, and only if `test` passed. It calls the Render deploy hook with `ref=<commit SHA>`, so exactly the tested commit is deployed.
3. **smoke-test** polls `GET /health` until it reports `status: ok` for that commit, then writes the result to the job summary.

Render auto-deploy is **off** (`render.yaml` and the dashboard), so the workflow is the only deploy path.

One-time setup:

| Where | Name | Value |
|---|---|---|
| Render → Environment | `ANTHROPIC_API_KEY` | your Anthropic key (secret) |
| Render → Settings | Auto-Deploy | **Off** |
| GitHub → Settings → Secrets → Actions | `RENDER_DEPLOY_HOOK_URL` | Render → Settings → Deploy Hook |
| GitHub → Settings → Variables → Actions | `APP_URL` | e.g. `https://malware-agent.onrender.com` |

## 6. Sample requests

Pre-filled examples (no typing of features needed):

```bash
APP=https://<your-app>.onrender.com
curl -s $APP/health
curl -s -X POST $APP/api/predict -H 'Content-Type: application/json' -d '{"example": "example_malware"}'
curl -s -X POST $APP/api/predict -H 'Content-Type: application/json' -d @<(jq '{record: .}' samples/single_record.json)
```

Batch and evaluation (keep the session cookie between calls):

```bash
curl -s -c jar -b jar -F file=@samples/unlabeled_batch.csv $APP/api/upload            # -> file_id
curl -s -c jar -b jar -X POST $APP/api/batch -H 'Content-Type: application/json' -d '{"file_id": "file_XXXX"}'
curl -s -c jar -b jar -F file=@samples/labeled_demo_mixed.csv $APP/api/upload
curl -s -c jar -b jar -X POST $APP/api/evaluate -H 'Content-Type: application/json' -d '{"file_id": "file_YYYY", "min_accuracy": 0.98}'
curl -s -c jar -b jar -X POST $APP/api/chat -H 'Content-Type: application/json' \
     -d '{"message": "Evaluate labeled_demo_mixed.csv; if accuracy is at least 90%, classify the example goodware record."}'
```

Example chat prompts:
* "Classify the pre-filled example malware record."
* "Classify every row in unlabeled_batch.csv." Then: "Which rows had the highest malware scores?"
* "Evaluate labeled_demo_mixed.csv. If accuracy is at least 98%, classify the example goodware record." Then: "How many false negatives were there?"

### Sample files (`samples/`)

| File | Contents |
|---|---|
| `single_record.json`, `single_record_goodware.json` | one complete feature record each (true malware / true goodware) |
| `unlabeled_batch.csv` | 50 rows, no Label |
| `batch_with_invalid_rows.csv` | 10 rows; row_4 has non-numeric Entropy, row_7 is missing NumberOfSections |
| `labeled_sample.csv` | 200 random hold-out rows (100 per class) |
| `labeled_demo_mixed.csv` | 100 labeled rows that **deliberately include 4 false negatives and 2 false positives** taken from the hold-out set, so the demo shows errors (accuracy 0.94, AUC 0.9839). It is not a random sample. |
| `labeled_single_class_malware.csv` | 30 malware-only rows (AUC undefined case) |

All samples were drawn from the hold-out partition **after** the final evaluation and were never used for training or tuning. None of them contain executable code; they hold feature values only.

## Environment variables

See [`.env.example`](.env.example): `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`, `SECRET_KEY`, `MAX_UPLOAD_MB`, `MAX_UPLOAD_ROWS`, `CHAT_RATE_LIMIT`, `MAX_TOOL_CALLS_PER_TURN`, `LLM_TIMEOUT_SECONDS`, `ENABLE_FAULT_INJECTION`, `SESSION_COOKIE_SECURE`.

## Citation

Dataset: F. Ceschin, F. Pinage, M. Castilho, D. Menotti, L. S. Oliveira and A. Gregio, "The Need for Speed: An Analysis of Brazilian Malware Classifiers," *IEEE Security & Privacy*, 16(6), 31–41, 2018. doi:10.1109/MSEC.2018.2875369.
