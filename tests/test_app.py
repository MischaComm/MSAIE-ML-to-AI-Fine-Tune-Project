"""Integration tests through the Flask application boundary."""

from __future__ import annotations

import io
import json

import pandas as pd

from conftest import upload


def test_health(make_app):
    _, c, _ = make_app()
    j = c.get("/health").get_json()
    assert j["status"] == "ok" and j["model_loaded"] and j["model_version"] == "1.0.0"


def test_index_page_loads(make_app):
    _, c, _ = make_app()
    r = c.get("/")
    assert r.status_code == 200 and b"Malware Analysis" in r.data


def test_predict_endpoint_with_example_and_inline_record(make_app):
    _, c, _ = make_app()
    r = c.post("/api/predict", json={"example": "example_malware"})
    assert r.status_code == 200 and r.get_json()["predicted_class"] == "malware"
    rec = json.load(open("samples/single_record_goodware.json"))
    r = c.post("/api/predict", json={"record": rec})
    assert r.get_json()["predicted_class"] == "goodware"


def test_predict_endpoint_malformed_and_missing_features(make_app):
    _, c, _ = make_app()
    rec = json.load(open("samples/single_record.json"))
    del rec["SizeOfCode"]
    rec["Entropy"] = "abc"
    r = c.post("/api/predict", json={"record": rec})
    assert r.status_code == 422
    err = r.get_json()["error"]
    assert "SizeOfCode" in err and "Entropy" in err


def test_batch_prediction_and_download(make_app):
    _, c, _ = make_app()
    fid = upload(c, "samples/batch_with_invalid_rows.csv").get_json()["file_id"]
    j = c.post("/api/batch", json={"file_id": fid}).get_json()
    assert j["n_rows_in_file"] == 10 and j["n_predicted"] == 8 and j["n_invalid"] == 2
    assert j["summary"]["malware"] + j["summary"]["goodware"] == 8
    dl = c.get(j["download_url"])
    df = pd.read_csv(io.BytesIO(dl.data))
    assert len(df) == 10  # invalid rows included, not dropped
    assert set(df.loc[df.status == "invalid", "row_id"]) == {"row_4", "row_7"}


def test_labeled_evaluation(make_app):
    _, c, _ = make_app()
    fid = upload(c, "samples/labeled_demo_mixed.csv").get_json()["file_id"]
    j = c.post("/api/evaluate", json={"file_id": fid, "min_accuracy": 0.9}).get_json()
    cm = j["confusion_matrix"]
    assert j["n_evaluated"] == 100
    assert cm["true_negatives"] + cm["false_positives"] + cm["false_negatives"] + cm["true_positives"] == 100
    assert j["metrics"]["accuracy"] == (cm["true_negatives"] + cm["true_positives"]) / 100
    assert j["accuracy_criterion"]["meets_min_accuracy"] is True
    assert j["label_mapping"] == {"0": "goodware", "1": "malware"}


def test_evaluation_missing_labels_and_single_class(make_app):
    _, c, _ = make_app()
    fid = upload(c, "samples/unlabeled_batch.csv").get_json()["file_id"]
    r = c.post("/api/evaluate", json={"file_id": fid})
    assert r.status_code == 422 and "labeled CSV" in r.get_json()["error"]
    fid = upload(c, "samples/labeled_single_class_malware.csv").get_json()["file_id"]
    j = c.post("/api/evaluate", json={"file_id": fid}).get_json()
    assert j["metrics"]["auc"] is None and "undefined" in j["metrics"]["auc_note"]
    assert j["metrics"]["accuracy"] is not None


def test_upload_validation(make_app):
    _, c, _ = make_app()
    r = c.post("/api/upload", data={"file": (io.BytesIO(b"hello"), "x.exe")}, content_type="multipart/form-data")
    assert r.status_code == 400
    r = c.post("/api/upload", data={"file": (io.BytesIO(b"a,b\n1,2\n"), "x.csv")}, content_type="multipart/form-data")
    assert r.status_code == 400 and "missing required feature" in r.get_json()["error"]
    big = b"x" * (6 * 1024 * 1024)
    r = c.post("/api/upload", data={"file": (io.BytesIO(big), "big.csv")}, content_type="multipart/form-data")
    assert r.status_code == 413


def test_sessions_are_isolated(make_app):
    app, c1, _ = make_app()
    c2 = app.test_client()
    fid = upload(c1, "samples/unlabeled_batch.csv").get_json()["file_id"]
    assert c2.get("/api/files").get_json() == []
    r = c2.post("/api/batch", json={"file_id": fid})
    assert r.status_code == 422 and "unknown file_id" in r.get_json()["error"]
    j = c1.post("/api/batch", json={"file_id": fid}).get_json()
    assert c2.get(j["download_url"]).status_code == 404
