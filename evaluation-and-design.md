# Evaluation and Design

Static malware detection on PE-file features: a trained ML model classifies records as **goodware (0)** or **malware (1)**, and an AI agent built on Anthropic Claude calls that model through tools. Every number below can be reproduced with `python train.py && python eval.py` (see README).

---

## 1. Dataset and problem definition

| Item | Value |
|---|---|
| Source | [brazilian-malware-dataset](https://github.com/fabriciojoc/brazilian-malware-dataset), commit `9f0d8a6`, file `brazilian-malware.zip` → `brazilian-malware.csv` |
| Checksums | zip `657136…a7b7`, csv `d13e54…9242` (full values in `malware_ml/schema.py`) |
| Raw size | 50,181 rows × 28 columns (27 attributes + `Label`) |
| Target | `Label` (column L): 0 = goodware, 1 = malware. Verified: values are exactly {0, 1}; 29,065 malware / 21,116 goodware |
| Primary metric | ROC AUC, computed from the model's continuous malware score (never from predicted labels) |
| Secondary metric | Accuracy at the frozen threshold 0.5 |

A prediction is a model assessment of static features. It does not establish that software is safe to run.

## 2. Data preparation decisions

**Order of operations (leakage control).**
1. Load the raw CSV and verify the checksum, columns and label mapping.
2. **Hold out 20% first.** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, taking the first fold as the test set, **grouped by SHA1**. Result: 40,144 training rows / 10,037 test rows (20.0%), with no SHA1 on both sides (asserted in code).
3. De-duplicate each partition separately with a deterministic rule that learns nothing from the data (below).
4. All learned preprocessing is fitted inside each CV training fold, and later on the full training partition only.

**Issues found (inspected on the training partition) and how each was handled.**

| Issue | Finding | Handling |
|---|---|---|
| Duplicate records | 6,770 raw rows repeat a SHA1. Every repeat has identical features, so these are re-observations of the same file on different days. Features map 1:1 to SHA1. | Grouped split (no file leaks across train/test), then keep one row per SHA1 in each partition. Train 40,144 → 34,716; test 10,037 → 8,677. Inside the training set every row is now a distinct file, so plain stratified K-fold cannot leak duplicates across folds. |
| Conflicting labels | 18 files appear with both labels (13 in train, 5 in test). | Dropped as ambiguous (38 + 16 rows). |
| Identifier column | `SHA1` | Not a model input; used only for grouping, de-duplication and display. |
| Collection metadata | `FirstSeenDate` records when a sample was observed, not a property of the executable. | Excluded to avoid temporal and collection-process leakage. |
| Missing values | Only `Identify` (packer/compiler signature) is missing: 10,381 training rows (30%). Missingness is informative because no signature matched. | Replaced with an explicit `__none__` token, so "no signature" becomes a feature. All other inputs are required. |
| Constant columns | `Magic`, `PE_TYPE`, `SizeOfOptionalHeader` are constant. | Removed by a `VarianceThreshold` fitted per training fold. |
| Heavy tails | Size and address fields span many orders of magnitude (max/median > 1000 for 6 columns). | Signed `log1p`, then `StandardScaler` (fitted per fold). |
| Class imbalance | Training set is 51.4% malware / 48.6% goodware after de-duplication. | Mild; no resampling. Stratified splits and folds preserve the ratio. |
| Leakage screen | Highest single-feature AUC is `DllCharacteristics` at 0.83. No single column is near-perfect. | No further action. |

**Final model inputs (25 columns).** 22 numeric PE header fields (3 of them dropped as constant inside the pipeline) plus 3 token-list fields: `Identify`, `ImportedDlls`, `ImportedSymbols`. Text fields become binary bag-of-tokens features with `CountVectorizer(binary=True, min_df=5)`, using the vocabularies of the top 50 / 300 / 1,000 tokens. The vocabulary is learned from the training fold only.

**Feature selection / dimensionality-reduction exploration** (10-fold CV, `reports/feature_ablation.csv`):

| Feature set | # features | LR AUC | LightGBM AUC |
|---|---|---|---|
| Numeric only | 19 | 0.9455 ± 0.0034 | 0.9977 ± 0.0006 |
| + Identify + ImportedDlls (top 50 / 300) | 369 | 0.9863 ± 0.0018 | 0.9984 ± 0.0005 |
| + ImportedSymbols top 1,000 (**chosen**) | 1,369 | 0.9939 ± 0.0015 | 0.9986 ± 0.0004 |
| Larger vocabularies (100 / 1,000 / 5,000) | 5,757 | 0.9948 ± 0.0010 | 0.9986 ± 0.0004 |

Justification: the text fields add a lot for linear and neural models and a little for boosted trees. Going from about 1.4k to about 5.8k features adds no AUC for the boosted trees, quadruples the feature count and slows tree training, so the frequency-capped vocabulary (top-k tokens by document frequency) is the selected reduction. Vocabulary capping is itself feature selection. Learned projection methods (such as TruncatedSVD on the token matrix) were not adopted: the capped sparse representation already performs at the ceiling, and it keeps features interpretable as named DLLs and symbols.

## 3. Evaluation protocol

* Stratified **10-fold CV** on the 34,716 training rows (`StratifiedKFold(10, shuffle=True, random_state=42)`).
* For each fold, the preprocessor is fitted on the 9 training folds and applied to the held-out fold. Fold matrices are cached and shared by all models, so every model sees identical splits.
* **Search strategy and scope:** a small, hand-specified grid per model (2–6 settings, listed in `malware_ml/models.py` and `reports/tuning_results.csv`). Each setting is screened on folds 1–3. The setting with the best mean screening AUC is then run on all 10 folds, and those 10-fold numbers are reported. This bounded the search to about 1 hour on a 2-core machine. Because the screening folds are part of the reported CV, the CV estimate carries a small optimistic bias. The untouched hold-out set is the unbiased check.
* AUC is computed from `predict_proba[:, 1]`; accuracy uses threshold 0.5.
* Seeds: 42 for the split, folds and all models (NumPy, PyTorch, library `random_state`).

## 4. Models and cross-validation results

10-fold CV, mean ± sample standard deviation (`reports/cv_results.csv`):

| Rank | Model | Group | CV AUC | CV Accuracy | Selected hyperparameters | Settings searched |
|---|---|---|---|---|---|---|
| 1 | **LightGBM** | additional | **0.9988 ± 0.0004** | **0.9883 ± 0.0017** | 400 trees, num_leaves 127, lr 0.05, subsample 0.8, colsample 0.5 | num_leaves {31, 127} × lr {0.05, 0.1} |
| 2 | XGBoost | additional | 0.9987 ± 0.0004 | 0.9878 ± 0.0017 | 400 trees, max_depth 10, lr 0.05, subsample 0.8, colsample 0.5 | max_depth {6, 10} × lr {0.05, 0.1} |
| 3 | CatBoost | additional | 0.9984 ± 0.0004 | 0.9870 ± 0.0019 | 500 iterations, depth 8, lr 0.1 | depth {6, 8} |
| 4 | Random Forest | baseline | 0.9983 ± 0.0004 | 0.9858 ± 0.0012 | 200 trees, max_features 0.05, min_samples_leaf 1 | max_features {sqrt, 0.05} × min_samples_leaf {1, 2} |
| 5 | PyTorch MLP | baseline | 0.9961 ± 0.0012 | 0.9787 ± 0.0022 | 2 hidden layers (256, 64), ReLU, dropout 0.2, Adam lr 1e-3, wd 1e-5, 15 epochs, batch 256 | {(256,64) d0.2 15ep, (512,128) d0.3 15ep, (256,64) d0.2 30ep wd1e-4} |
| 6 | Logistic Regression | baseline | 0.9942 ± 0.0013 | 0.9712 ± 0.0029 | L2, C = 0.1, liblinear | C {0.1, 1, 10} |
| 7 | Decision Tree | baseline | 0.9891 ± 0.0016 | 0.9744 ± 0.0027 | max_depth 10, min_samples_leaf 5 | max_depth {10, 20, None} × min_samples_leaf {1, 5} |

Mean fit time per fold / scoring time per 1,000 rows: LightGBM 15.7 s / 24 ms, XGBoost 3.9 s / 12 ms, CatBoost 62.5 s / 6 ms, Random Forest 57.3 s / 33 ms, MLP 7.5 s / 4 ms, LR 1.3 s / 0.7 ms, Decision Tree 1.4 s / 0.5 ms.

**Top three:** LightGBM, XGBoost, CatBoost. **Bottom two:** Logistic Regression (lowest AUC among the remaining models), Decision Tree (lowest overall). Decision Tree has slightly higher accuracy than Logistic Regression but a much lower AUC: a single depth-10 tree outputs only as many distinct scores as it has leaves, so its ranking (AUC) is coarse even when its 0.5 cut is reasonable.

## 5. Production model selection

**Rule (fixed before looking at the hold-out set):** highest mean CV AUC. Models within 0.001 AUC of the best are ranked by mean CV accuracy, then by inference speed.

LightGBM, XGBoost, CatBoost and Random Forest all fall within 0.001 AUC. Ranked by accuracy, **LightGBM** (0.9883) leads XGBoost (0.9878), so LightGBM was selected. Production considerations support the choice: the model is 2.4 MB, scoring a single record takes milliseconds, and LightGBM needs no GPU or heavy runtime on a 512 MB free-tier instance. XGBoost would be an equally defensible choice: statistically it is a tie, and it scores about twice as fast.

**Frozen before testing:** preprocessing configuration, hyperparameters and threshold 0.5. The threshold is the default, not tuned. Model version `1.0.0`. The pipeline was then refitted on all 34,716 training rows (20.6 s) and saved as `artifacts/model.joblib`, together with `artifacts/metadata.json` (schema, label mapping, threshold, version, CV scores, library versions).

## 6. Hold-out test results (evaluated once)

`python eval.py` on the 8,677 de-duplicated hold-out files (`reports/holdout_results.json`):

| Metric | Value |
|---|---|
| **AUC** | **0.998916** |
| **Accuracy** | **0.989282** (98.93%) |

Confusion matrix (rows = true label, columns = predicted label; positive class = malware):

| | Predicted goodware (0) | Predicted malware (1) |
|---|---|---|
| **True goodware (0)** — 4,219 | 4,178 (TN) | 41 (FP) |
| **True malware (1)** — 4,458 | 52 (FN) | 4,406 (TP) |

The hold-out results match CV (0.9988 AUC / 0.9883 accuracy), which is consistent with no leakage and no overfitting to the CV folds. The test set was not used for any tuning or selection. After this evaluation, hold-out rows were reused only to build the small demo CSVs in `samples/`.

## 7. Agent and overall architecture

```
Browser (chat UI + CSV/JSON upload)
   │  HTTPS, session cookie (random id, signed)
   ▼
Flask app (gunicorn, 1 worker × 4 threads)          ← /health, /api/upload, /api/chat,
   │                                                  /api/predict|batch|evaluate, /api/results/<id>/download
   ├── SessionStore: per-session upload dir, result objects, chat history (in memory, 2 h TTL)
   ├── Agent loop (app/agent.py)
   │      └── Anthropic Messages API (Claude, tool use) ← API key is a server-side secret
   │             decides which tool to call and with which arguments
   └── Tools (app/tools.py) ──► ProductionModel (malware_ml/inference.py)
                                   └── artifacts/model.joblib (preprocessing + LightGBM) + metadata.json
```

* **The ML model makes every classification.** Tools call `ProductionModel`, which applies the saved pipeline and threshold. The system prompt forbids the AI model from judging files itself or inventing numbers.
* **Bounded context.** Uploaded files stay on the server. The AI model sees only file ids, names, row counts and whether a `Label` column exists. Tools return summaries: counts, metrics, a 10-row preview and at most 50 rows per follow-up query. Raw feature text is never sent. Row-level results are stored server-side and reached through `query_results`.
* **Conditional prediction.** `evaluate_labeled_data` accepts `min_accuracy`. Code computes `meets_min_accuracy = accuracy >= min_accuracy`. The agent reads that returned value and calls `predict_single_record` only if it is true. The activity record shows the sequence.
* **Activity record.** Each reply carries the tools called, their arguments, their status, and a short result summary or error. The UI displays this under every answer, and the server logs tool name and status (never credentials).
* **Failure handling.** Tool errors go back to the AI model as `is_error` results, appear as errors in the activity record, and add a warning banner that code generates independently of the AI model's wording. LLM timeouts or API errors return a clear 502 message with no partial result.

### Tool schemas

| Tool | Arguments | Returns (to the AI model) |
|---|---|---|
| `predict_single_record` (Single Record Prediction Tool) | exactly one of `example` ∈ {`example_malware`, `example_goodware`}, `file_id` (+ optional `row_id`), `record` (object) | `predicted_label`, `predicted_class`, `malware_score` (uncalibrated), `threshold`, `model_version`, `result_id` |
| `predict_batch` (Batch Prediction Tool) | `file_id` | `n_rows_in_file`, `n_predicted`, `n_invalid`, `summary` {malware, goodware, invalid_not_scored}, 10-row preview, invalid rows with reasons, `download_url`, `result_id` |
| `evaluate_labeled_data` (Labeled Data Evaluation Tool) | `file_id`, optional `min_accuracy` (fraction) | `n_evaluated`, `n_invalid`, `true_class_counts`, `predicted_class_counts`, `label_mapping`, `metrics` {auc, accuracy, auc_note}, `confusion_matrix` {TN, FP, FN, TP + labeled layout}, `accuracy_criterion`, `result_id` |
| `query_results` (follow-ups) | `result_id`, `query` ∈ {false_negatives, false_positives, true_positives, true_negatives, predicted_malware, predicted_goodware, invalid_rows, row, highest_scores, lowest_scores}, `row_id`, `limit` ≤ 50 | `count` and the matching rows (row_id, true/predicted label, score) |

Full JSON schemas: `TOOL_SCHEMAS` in `app/tools.py`.

**Validation.** Records are checked against the saved schema: all 25 features present except the nullable `Identify`, numerics finite, text fields strings, unknown keys rejected. `Label`, `SHA1` and `FirstSeenDate` are ignored if supplied. In batches, invalid rows are listed with row ids and reasons, never silently dropped. Evaluation requires a `Label` column with values in {0, 1}. With only one true class present, AUC is returned as `null` with an explanation, and accuracy and the confusion matrix are still reported.

**Row identifiers.** `row_N` is the 1-based data-row number in the uploaded file, header excluded. It is stable for a given file. The SHA1 is echoed when present.

## 8. AI model choice

Anthropic **Claude Sonnet 5.5** (`claude-sonnet-5-5`), set by the `ANTHROPIC_MODEL` environment variable. I chose it for reliable multi-step tool use (the conditional evaluate-then-predict sequence), faithful reporting of numbers, and adequate latency for a chat UI. A smaller model (Claude Haiku 5.5) can be configured to lower cost. In either case the ML model, not the AI model, produces every classification.

## 9. Security and operational limits

* API key and Flask secret are server-side environment variables (Render secrets). They are never in the repo or the browser, and the grader needs no key.
* Uploads: `.csv` / `.json` only, ≤ 5 MB, ≤ 20,000 rows, required feature columns checked at upload. Files are parsed as data and never executed. File contents and names are treated as data; the system prompt tells the agent to ignore instructions inside them.
* The agent can call only the 4 project tools. There is no shell or code execution, and unknown tool names are rejected.
* Limits: 6 tool calls per request, 4,000-character messages, 20 agent requests per 10 minutes per session, 60 s LLM timeout with 2 retries, 120 s gunicorn timeout, one concurrent agent request per session.
* Sessions: a random id in an HttpOnly, SameSite=Lax, Secure cookie. Each session gets its own upload directory and result store, and requests for another session's file or result id return "not found". Data expires after 2 hours.

**Operational limitations**
* Render free tier: the service sleeps after about 15 minutes idle, so the first request can take about 1 minute. It has 512 MB RAM and builds take several minutes.
* Session data lives in memory on one instance and is lost on restart or redeploy. The design needs a single worker process; horizontal scaling would need a shared store such as Redis.
* `malware_score` is not calibrated. There is no feature-attribution tool, so the agent cannot explain why an individual prediction was made.
* The dataset covers Brazilian samples collected 2010s–2020, so performance on newer or different malware families is not established (concept drift).
* AI-model answers can vary between runs. CI uses a mocked model for determinism, and live behaviour is evaluated separately in `agent-evaluation.md`.
