# Agent Evaluation with the Actual AI Model

> **Status: to be completed after deployment.** Run
> `python scripts/run_agent_eval.py --url https://<live-app>.onrender.com`,
> review each reply in `reports/agent_eval_results.json`, then fill in the tables below.
> The scenario design, expected behaviour and checks are final.

These scenarios use the **deployed** app and the **real** configured AI model. They are separate from the mocked agent tests in CI (`tests/test_agent.py`).

| Item | Value |
|---|---|
| AI model | `claude-sonnet-5-5` (from `/health` → `llm_model`) |
| ML model | LightGBM v1.0.0, threshold 0.5 |
| App URL / commit | _fill in_ |
| Evaluation date | _fill in_ |
| Method | `scripts/run_agent_eval.py`: each scenario starts a fresh browser session (except the follow-up, which reuses scenario 3b's session), uploads sample files through `/api/upload`, sends the request to `/api/chat`, and records the activity record and reply. Numerical statements in replies are checked automatically against the tool outputs, then reviewed by hand. |

## Scenarios

| # | Scenario | Request | Expected behaviour |
|---|---|---|---|
| 1 | Single prediction | "Classify the pre-filled example malware record." | `predict_single_record(example="example_malware")`; reports the model's class, score (described as uncalibrated) and version |
| 2 | Batch prediction | upload `unlabeled_batch.csv`; "Classify every row … give me the malware and goodware counts." | `predict_batch(file_id)`; counts match the tool summary; link to per-row results |
| 3a | Evaluation → prediction **permitted** | upload `labeled_demo_mixed.csv`; "Evaluate … If the accuracy is at least 90%, classify the pre-filled example goodware record …" | `evaluate_labeled_data(min_accuracy=0.9)` → accuracy 0.94 ≥ 0.90 → `predict_single_record` called and reported |
| 3b | Evaluation → prediction **withheld** | same file; threshold 98% | `evaluate_labeled_data(min_accuracy=0.98)` → 0.94 < 0.98 → **no** `predict_single_record`; agent explains why |
| 4 | False-negative follow-up | (same session as 3b) "How many false negatives were there … which rows?" | answer = 4 (confusion-matrix FN), from the stored result, optionally via `query_results(false_negatives)`; FN defined as true malware predicted goodware |
| 5 | Invalid input | upload `batch_with_invalid_rows.csv`; "Classify …" | `predict_batch`; 8 rows scored; row_4 (non-numeric Entropy) and row_7 (missing NumberOfSections) reported as invalid, not dropped |
| 6 | Missing labels | upload `unlabeled_batch.csv`; "Evaluate the model's accuracy on it." | no metrics; explains that there is no Label column and asks for a labeled CSV |
| 7 | Single-class evaluation | upload `labeled_single_class_malware.csv`; "Evaluate …" | `evaluate_labeled_data`; AUC reported as undefined (one true class), accuracy and confusion matrix still reported |
| 8 | Tool / service failure | "Classify the pre-filled example malware record." with the *simulate tool failure* control (server flag `ENABLE_FAULT_INJECTION`) | tool returns an error; agent says the step failed and reports no prediction; UI shows a warning banner |

## Results

_Paste `reports/agent_eval_table.md` here and add your manual review notes._

| # | Tools actually called (arguments) | Observed result | Numerical check | Pass/Fail |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3a | | | | |
| 3b | | | | |
| 4 | | | | |
| 5 | | | | |
| 6 | | | | |
| 7 | | | | |
| 8 | | | | |

**Scenario pass rate:** _x / 9_ (8 required scenarios; the conditional scenario is run in both its permitted and withheld forms).

## Failure patterns and findings

_Summarise any failures or near-misses (e.g. rounding of percentages, extra tool calls, wording that implies calibrated confidence), what caused them, and any prompt or tool changes made in response._
