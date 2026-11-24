# AI Tooling Used in Development

> Draft: edit this into your own words before submitting.

These are the development tools. They are separate from the AI model that powers the deployed agent (Claude Sonnet 5.5, called at runtime through the Anthropic API).

## Tools

* **Claude (Anthropic) in an agentic coding session.** Claude worked in a cloud Linux workspace with a shell, file editing and web search. I supplied the project requirements PDF and made the decisions about scope, the AI provider and hosting.

## How they were used

* **Requirements analysis.** Claude read the project brief and rubric and summarised the deliverables, scoring criteria and minimum passing bar.
* **Data inspection.** Claude profiled the dataset and surfaced the issues that shaped the protocol: duplicate SHA1 records, conflicting labels, constant columns, `FirstSeenDate` as possible leakage, and the missing-value pattern in `Identify`. I reviewed and approved each handling decision.
* **ML pipeline code.** Claude generated `malware_ml/` (schema, preprocessing, PyTorch MLP wrapper, model grids, inference), `train.py` and `eval.py`, and ran the full 10-fold CV experiment and hold-out evaluation.
* **Agent and app.** Claude generated the Flask app, the tool layer, the Claude tool-use loop, the chat UI, the CI/CD workflow, the Render configuration and the documentation drafts.
* **Testing.** Claude wrote unit, integration and mocked-AI agent tests, plus a script that runs the live evaluation scenarios against the deployed app.

## What worked well

* **Speed.** A full, reproducible pipeline with seven models and a deployed agent was built in a fraction of the time hand-coding would take.
* **Data checks.** Claude flagged leakage risks (duplicate files across partitions, collection-date metadata) before training rather than after.
* **Tests that found real bugs.** For example, writing the test harness revealed that text the agent produced alongside a tool call was being dropped from replies. Claude also caught an arithmetic error in one of its own hand-computed AUC test values.

## What worked less well / required oversight

* **Environment limits.** The workspace's network allowlist blocked direct downloads from GitHub's raw-file host, so the dataset download script could not be tested there. It was verified on my machine / in CI instead.
* **Library-version surprises.** A pandas 3 behaviour change broke a sampling snippet, and a dense-vs-sparse matrix mismatch broke the first full training run. Both were fixed quickly, but they show that generated code must be run, not just read.
* **Repository access.** Connecting GitHub to the coding session took several attempts.
* **I remain responsible for the result.** I reviewed the design decisions, the numbers in the documentation (all of them come from files in `reports/`), and the agent's live behaviour in `agent-evaluation.md`.
