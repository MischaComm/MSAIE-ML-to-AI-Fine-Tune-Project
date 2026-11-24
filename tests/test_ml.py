"""Unit tests: schema validation, preprocessing, model wrapper, class mapping, metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from malware_ml import schema
from malware_ml.features import build_preprocessor, deduplicate, validate_record
from malware_ml.inference import ProductionModel, ValidationError, get_model

SAMPLE = pd.read_csv("samples/labeled_sample.csv")


def good_record():
    return SAMPLE.drop(columns=["Label"]).iloc[0].to_dict()


# ----------------------------------------------------------------- schema / validation
def test_label_mapping():
    assert schema.LABEL_MAPPING == {0: "goodware", 1: "malware"}
    assert schema.POSITIVE_CLASS == 1
    assert schema.LABEL_COL not in schema.MODEL_FEATURES


def test_valid_record_passes():
    r = validate_record(good_record())
    assert r.ok, r.errors


def test_missing_feature_rejected():
    rec = good_record()
    del rec["Entropy"]
    r = validate_record(rec)
    assert not r.ok and any("Entropy" in e for e in r.errors)


def test_wrong_type_rejected():
    rec = good_record()
    rec["NumberOfSections"] = "lots"
    rec["ImportedDlls"] = 42
    r = validate_record(rec)
    assert not r.ok
    assert any("NumberOfSections" in e for e in r.errors)
    assert any("ImportedDlls" in e for e in r.errors)


def test_unknown_feature_rejected():
    rec = good_record()
    rec["Bogus"] = 1
    assert not validate_record(rec).ok


def test_nullable_identify_allowed_and_label_ignored():
    rec = good_record()
    rec["Identify"] = None
    rec["Label"] = 1  # must be ignored, never a model input
    r = validate_record(rec)
    assert r.ok
    assert "Label" not in r.clean


def test_numeric_string_coerced_and_infinite_rejected():
    rec = good_record()
    rec["Entropy"] = "6.5"
    assert validate_record(rec).clean["Entropy"] == 6.5
    rec["Entropy"] = float("inf")
    assert not validate_record(rec).ok


# ----------------------------------------------------------------- preprocessing
def test_preprocessor_fits_only_on_given_data_and_drops_constants():
    df = SAMPLE.copy()
    pre = build_preprocessor(n_ident=10, n_dll=20, n_sym=30)
    Xt = pre.fit_transform(df[schema.MODEL_FEATURES])
    assert Xt.shape[0] == len(df)
    var = pre.named_transformers_["num"].named_steps["var"]
    kept = set(np.array(schema.NUMERIC_FEATURES)[var.get_support()])
    # Magic, PE_TYPE and SizeOfOptionalHeader are constant in the data
    assert {"Magic", "PE_TYPE", "SizeOfOptionalHeader"}.isdisjoint(kept)


def test_deduplicate_removes_repeat_files_and_conflicts():
    df = pd.DataFrame({"SHA1": ["a", "a", "b", "b", "c"], "Label": [1, 1, 0, 1, 0], "x": [1, 1, 2, 2, 3]})
    out, stats = deduplicate(df)
    assert sorted(out.SHA1) == ["a", "c"]
    assert stats["conflicting_label_files"] == 1


# ----------------------------------------------------------------- model wrapper
def test_predict_record_uses_threshold_and_mapping():
    m = get_model()
    out = m.predict_record(good_record())
    assert out["predicted_label"] == int(out["malware_score"] >= m.threshold)
    assert out["predicted_class"] == schema.LABEL_MAPPING[out["predicted_label"]]
    assert out["model_version"] == m.version


def test_predict_record_invalid_raises():
    with pytest.raises(ValidationError):
        get_model().predict_record({"Entropy": 1})


def test_batch_reports_invalid_rows_not_silently_dropped():
    df = SAMPLE.drop(columns=["Label"]).head(5).copy()
    df["Entropy"] = df["Entropy"].astype(object)
    df.loc[2, "Entropy"] = "not-a-number"
    out = get_model().predict_frame(df)
    assert out["n_rows_in_file"] == 5
    assert out["n_predicted"] == 4
    assert out["invalid_rows"][0]["row_id"] == "row_3"
    assert out["summary"]["malware"] + out["summary"]["goodware"] == 4


# ----------------------------------------------------------------- metrics
class _StubPipe:
    """Returns fixed scores so metrics can be checked against hand-computed values."""

    def __init__(self, scores):
        self.scores = np.asarray(scores)

    def predict_proba(self, X):
        s = self.scores[: len(X)]
        return np.column_stack([1 - s, s])


def _stub_model(scores):
    meta = {"classification_threshold": 0.5, "model_version": "test", "model_name": "stub"}
    return ProductionModel(_StubPipe(scores), meta)


def test_metrics_against_known_values():
    df = SAMPLE.head(6).copy()
    df["Label"] = [0, 0, 0, 1, 1, 1]
    # predictions:  0    1    0    1    0    1   -> TN=2 FP=1 FN=1 TP=2
    m = _stub_model([0.1, 0.6, 0.2, 0.9, 0.4, 0.8])
    out = m.evaluate_frame(df)
    cm = out["confusion_matrix"]
    assert (cm["true_negatives"], cm["false_positives"], cm["false_negatives"], cm["true_positives"]) == (2, 1, 1, 2)
    assert out["metrics"]["accuracy"] == pytest.approx(4 / 6, abs=1e-6)
    # AUC by hand: positives {0.9,0.4,0.8} vs negatives {0.1,0.6,0.2}:
    # 0.9 beats 3, 0.4 beats 2 (0.1, 0.2), 0.8 beats 3 -> 8 of 9 pairs ordered
    assert out["metrics"]["auc"] == pytest.approx(8 / 9, abs=1e-6)


def test_single_class_auc_undefined():
    df = SAMPLE.head(4).copy()
    df["Label"] = 1
    out = _stub_model([0.9, 0.8, 0.3, 0.7]).evaluate_frame(df)
    assert out["metrics"]["auc"] is None
    assert "undefined" in out["metrics"]["auc_note"]
    assert out["metrics"]["accuracy"] == pytest.approx(0.75)


def test_missing_labels_rejected():
    with pytest.raises(ValidationError, match="labeled CSV"):
        get_model().evaluate_frame(SAMPLE.drop(columns=["Label"]))
