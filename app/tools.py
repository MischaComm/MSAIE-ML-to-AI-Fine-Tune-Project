"""Agent tools. Every classification and metric comes from the production ML model
(malware_ml.inference). Tools receive file *references*, read the files on the
server, and return bounded summaries to the AI model — never whole CSVs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from malware_ml.inference import ValidationError, get_model
from malware_ml.schema import LABEL_COL

from .store import Session

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = {
    "example_malware": ROOT / "samples" / "single_record.json",
    "example_goodware": ROOT / "samples" / "single_record_goodware.json",
}
PREVIEW_ROWS = 10
MAX_QUERY_ROWS = 50


class ToolError(Exception):
    """A tool failed; the message is safe to show to the AI model and the user."""


# --------------------------------------------------------------------------- schemas (Anthropic format)
TOOL_SCHEMAS = [
    {
        "name": "predict_single_record",
        "description": (
            "Single Record Prediction Tool. Classifies ONE PE-file feature record as malware or goodware "
            "using the production ML model. Provide exactly one source: `example` (a pre-filled sample record), "
            "`file_id` (an uploaded .json record, or an uploaded CSV plus `row_id` such as 'row_1'), or "
            "`record` (an object with all feature values, e.g. pasted by the user). Returns predicted class, "
            "uncalibrated malware score, threshold and model version. Any Label value in the input is ignored."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "example": {"type": "string", "enum": list(EXAMPLES), "description": "Use a pre-filled example record."},
                "file_id": {"type": "string", "description": "Id of an uploaded file in this session."},
                "row_id": {"type": "string", "description": "Row in an uploaded CSV, e.g. 'row_1' (first data row). Defaults to row_1 for single-row files."},
                "record": {"type": "object", "description": "Feature name -> value for one record."},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "predict_batch",
        "description": (
            "Batch Prediction Tool. Classifies every row of an uploaded CSV with the production ML model. "
            "Returns malware/goodware counts, invalid-row count with reasons (invalid rows are listed, never "
            "silently dropped), a short preview, a result_id for follow-up queries, and a download link to "
            "the full per-row results. A Label column, if present, is ignored."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"file_id": {"type": "string", "description": "Id of an uploaded CSV in this session."}},
            "required": ["file_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "evaluate_labeled_data",
        "description": (
            "Labeled Data Evaluation Tool. Scores an uploaded CSV that contains the features AND true Label "
            "values (0 = goodware, 1 = malware) with the production ML model and computes AUC, accuracy and "
            "the confusion matrix in code. If `min_accuracy` is given (fraction 0-1, e.g. 0.98), the result "
            "also states whether accuracy >= min_accuracy. Returns a result_id for follow-up queries."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_id": {"type": "string", "description": "Id of an uploaded labeled CSV in this session."},
                "min_accuracy": {"type": "number", "description": "Optional accuracy criterion as a fraction (0.98 for 98%)."},
            },
            "required": ["file_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "query_results",
        "description": (
            "Look up rows from an earlier batch or evaluation result in this session, for follow-up questions "
            "(e.g. which rows were false negatives, the prediction for row_12). Counts come from the stored "
            "result, not from estimation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "result_id": {"type": "string"},
                "query": {
                    "type": "string",
                    "enum": [
                        "false_negatives", "false_positives", "true_positives", "true_negatives",
                        "predicted_malware", "predicted_goodware", "invalid_rows", "row", "highest_scores", "lowest_scores",
                    ],
                },
                "row_id": {"type": "string", "description": "Required when query = 'row'."},
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_QUERY_ROWS},
            },
            "required": ["result_id", "query"],
            "additionalProperties": False,
        },
    },
]


# --------------------------------------------------------------------------- helpers
def _get_file(session: Session, file_id) -> object:
    if not file_id or not isinstance(file_id, str):
        raise ToolError("file_id is required")
    f = session.files.get(file_id)
    if f is None:
        known = ", ".join(session.files) or "none"
        raise ToolError(f"unknown file_id '{file_id}'. Files in this session: {known}")
    return f


def _read_csv(path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as e:  # malformed CSV
        raise ToolError(f"could not parse CSV: {str(e)[:200]}")


# --------------------------------------------------------------------------- tools
def predict_single_record(session: Session, args: dict) -> tuple[dict, dict | None]:
    sources = [k for k in ("example", "file_id", "record") if args.get(k) not in (None, "", {})]
    if len(sources) != 1:
        raise ToolError("provide exactly one of: example, file_id, record")
    src = sources[0]
    if src == "example":
        if args["example"] not in EXAMPLES:
            raise ToolError(f"example must be one of {list(EXAMPLES)}")
        record = json.loads(EXAMPLES[args["example"]].read_text())
        source_desc = args["example"]
    elif src == "record":
        record = args["record"]
        source_desc = "inline record"
    else:
        f = _get_file(session, args["file_id"])
        if f.kind == "json":
            record = json.loads(f.path.read_text())
            if isinstance(record, list):
                if len(record) != 1:
                    raise ToolError("JSON file must contain one record object")
                record = record[0]
            source_desc = f.name
        else:
            df = _read_csv(f.path)
            row_id = args.get("row_id") or ("row_1" if len(df) == 1 else None)
            if row_id is None:
                raise ToolError(f"{f.name} has {len(df)} rows; specify row_id (row_1 .. row_{len(df)}) or use predict_batch")
            try:
                n = int(str(row_id).replace("row_", ""))
                assert 1 <= n <= len(df)
            except (ValueError, AssertionError):
                raise ToolError(f"row_id must be between row_1 and row_{len(df)}")
            record = {k: (None if pd.isna(v) else v) for k, v in df.iloc[n - 1].to_dict().items()}
            source_desc = f"{f.name} {row_id}"
    try:
        out = get_model().predict_record(record)
    except ValidationError as e:
        raise ToolError("record failed schema validation: " + "; ".join(e.errors[:10]))
    out = {"source": source_desc, **out}
    if isinstance(record, dict) and LABEL_COL in record:
        out["note"] = "the input contained a Label value; it was ignored by the model"
    rid = session.add_result("pred", {"result": out})
    return {"result_id": rid, **out}, None


def predict_batch(session: Session, args: dict) -> tuple[dict, dict | None]:
    f = _get_file(session, args.get("file_id"))
    if f.kind != "csv":
        raise ToolError("predict_batch needs an uploaded CSV file")
    df = _read_csv(f.path)
    if len(df) == 0:
        raise ToolError("the CSV has no data rows")
    try:
        full = get_model().predict_frame(df.drop(columns=[LABEL_COL], errors="ignore"))
    except ValueError as e:
        raise ToolError(str(e))
    rid = session.add_result("batch", {"file": f.name, "result": full})
    llm = {k: v for k, v in full.items() if k not in ("predictions", "invalid_rows")}
    llm.update({
        "result_id": rid,
        "file": f.name,
        "preview_first_rows": full["predictions"][:PREVIEW_ROWS],
        "invalid_rows": full["invalid_rows"][:20],
        "download_url": f"/api/results/{rid}/download",
    })
    return llm, {"download_url": llm["download_url"]}


def evaluate_labeled_data(session: Session, args: dict) -> tuple[dict, dict | None]:
    f = _get_file(session, args.get("file_id"))
    if f.kind != "csv":
        raise ToolError("evaluate_labeled_data needs an uploaded CSV file")
    df = _read_csv(f.path)
    if LABEL_COL not in df.columns or df[LABEL_COL].isna().all():
        raise ToolError(
            f"{f.name} has no true '{LABEL_COL}' values, so it cannot be evaluated. "
            "Please upload a labeled CSV (Label: 0 = goodware, 1 = malware)."
        )
    try:
        full = get_model().evaluate_frame(df)
    except ValidationError as e:
        raise ToolError("; ".join(e.errors))
    rid = session.add_result("eval", {"file": f.name, "result": full})
    llm = {k: v for k, v in full.items() if not k.startswith("_")}
    llm.update({"result_id": rid, "file": f.name})
    min_acc = args.get("min_accuracy")
    if min_acc is not None:
        try:
            min_acc = float(min_acc)
        except (TypeError, ValueError):
            raise ToolError("min_accuracy must be a number")
        if min_acc > 1:  # tolerate "98" meaning 98%
            min_acc = min_acc / 100
        acc = full["metrics"]["accuracy"]
        llm["accuracy_criterion"] = {
            "min_accuracy": round(min_acc, 6),
            "observed_accuracy": acc,
            "meets_min_accuracy": bool(acc is not None and acc >= min_acc),
            "comparison": "observed_accuracy >= min_accuracy",
        }
    return llm, None


def query_results(session: Session, args: dict) -> tuple[dict, dict | None]:
    rid = args.get("result_id")
    res = session.results.get(rid)
    if res is None:
        raise ToolError(f"unknown result_id '{rid}'. Available: {', '.join(session.results) or 'none'}")
    q = args.get("query")
    limit = min(int(args.get("limit") or 20), MAX_QUERY_ROWS)
    data = res["result"]
    if res["kind"] == "eval":
        rows = data["_row_results"]
    elif res["kind"] == "batch":
        rows = data["predictions"]
    else:
        return {"result_id": rid, "result": data}, None

    def has_truth():
        if res["kind"] != "eval":
            raise ToolError(f"'{q}' needs true labels; {rid} is a batch prediction without labels")

    filters = {
        "false_negatives": lambda r: (has_truth(), r["true_label"] == 1 and r["predicted_label"] == 0)[1],
        "false_positives": lambda r: (has_truth(), r["true_label"] == 0 and r["predicted_label"] == 1)[1],
        "true_positives": lambda r: (has_truth(), r["true_label"] == 1 and r["predicted_label"] == 1)[1],
        "true_negatives": lambda r: (has_truth(), r["true_label"] == 0 and r["predicted_label"] == 0)[1],
        "predicted_malware": lambda r: r["predicted_label"] == 1,
        "predicted_goodware": lambda r: r["predicted_label"] == 0,
    }
    if q in filters:
        matched = [r for r in rows if filters[q](r)]
        return {"result_id": rid, "query": q, "count": len(matched), "rows": matched[:limit], "rows_truncated": len(matched) > limit}, None
    if q in ("highest_scores", "lowest_scores"):
        s = sorted(rows, key=lambda r: r["malware_score"], reverse=q == "highest_scores")
        return {"result_id": rid, "query": q, "rows": s[:limit]}, None
    if q == "invalid_rows":
        inv = data.get("invalid_rows", [])
        return {"result_id": rid, "query": q, "count": data.get("n_invalid", len(inv)), "rows": inv[:limit]}, None
    if q == "row":
        row_id = args.get("row_id")
        for r in rows:
            if r["row_id"] == row_id:
                return {"result_id": rid, "query": q, "row": r}, None
        for r in data.get("invalid_rows", []):
            if r["row_id"] == row_id:
                return {"result_id": rid, "query": q, "row": r, "note": "this row was invalid and was not scored"}, None
        raise ToolError(f"row_id '{row_id}' not found in {rid}")
    raise ToolError(f"unsupported query '{q}'")


TOOLS = {
    "predict_single_record": predict_single_record,
    "predict_batch": predict_batch,
    "evaluate_labeled_data": evaluate_labeled_data,
    "query_results": query_results,
}
