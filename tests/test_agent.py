"""Agent tests with a scripted (mocked) AI model: tool dispatch, argument
validation, result handling, conditional prediction and failure behaviour."""

from __future__ import annotations

import json

from conftest import last_tool_results, response, text_block, tool_block, upload


def chat(c, msg, **kw):
    return c.post("/api/chat", json={"message": msg, **kw})


def test_single_record_dispatch(make_app):
    steps = [
        lambda m: response(tool_block("predict_single_record", {"example": "example_malware"})),
        lambda m: response(text_block(f"Predicted: {last_tool_results(m)[0]['predicted_class']}")),
    ]
    _, c, llm = make_app(steps)
    j = chat(c, "classify the example malware record").get_json()
    assert j["status"] == "ok"
    assert [a["tool"] for a in j["activity"]] == ["predict_single_record"]
    assert j["activity"][0]["status"] == "ok"
    assert j["reply"] == "Predicted: malware"
    assert j["tool_errors"] == 0
    # the system prompt and the four project tools are sent to the model
    assert {t["name"] for t in llm.calls[0]["tools"]} == {"predict_single_record", "predict_batch", "evaluate_labeled_data", "query_results"}


def test_batch_dispatch_sends_bounded_summary_not_csv(make_app):
    _, c, llm = make_app()
    fid = upload(c, "samples/unlabeled_batch.csv").get_json()["file_id"]
    llm.steps = [
        lambda m: response(tool_block("predict_batch", {"file_id": fid})),
        lambda m: response(text_block("done")),
    ]
    j = chat(c, "classify my batch").get_json()
    assert j["activity"][0]["result"]["n_predicted"] == 50
    assert j["links"][0]["url"].startswith("/api/results/")
    sent = json.dumps(llm.calls[1]["messages"])
    # raw feature text (imported symbols) never reaches the model; only ids/summaries
    assert "getprocaddress" not in sent.lower()
    assert len(last_tool_results(llm.calls[1]["messages"])[0]["preview_first_rows"]) == 10


def _conditional_policy(record_example="example_goodware"):
    """Mock model that follows the system prompt: decide using meets_min_accuracy."""

    def decide(m):
        res = last_tool_results(m)[0]
        if res["status"] == "ok" and res["accuracy_criterion"]["meets_min_accuracy"]:
            return response(tool_block("predict_single_record", {"example": record_example}, "tu_2"))
        return response(text_block("No prediction was made because the accuracy criterion was not met."))

    return decide


def _run_conditional(make_app, min_acc):
    app, c, llm = make_app()
    fid = upload(c, "samples/labeled_demo_mixed.csv").get_json()["file_id"]
    llm.steps = [
        lambda m: response(tool_block("evaluate_labeled_data", {"file_id": fid, "min_accuracy": min_acc})),
        _conditional_policy(),
        lambda m: response(text_block("Prediction made.")),
    ]
    return chat(c, f"evaluate and predict if accuracy >= {min_acc}").get_json(), llm


def test_conditional_prediction_permitted(make_app):
    j, _ = _run_conditional(make_app, 0.90)  # demo file accuracy is 0.94
    assert [a["tool"] for a in j["activity"]] == ["evaluate_labeled_data", "predict_single_record"]
    assert j["activity"][0]["result"]["accuracy_criterion"]["meets_min_accuracy"] is True


def test_text_alongside_tool_calls_is_kept_in_reply(make_app):
    steps = [
        lambda m: response(text_block("Step one done."), tool_block("predict_single_record", {"example": "example_malware"})),
        lambda m: response(text_block("Step two done.")),
    ]
    _, c, _ = make_app(steps)
    assert chat(c, "go").get_json()["reply"] == "Step one done.\n\nStep two done."


def test_conditional_prediction_withheld(make_app):
    j, llm = _run_conditional(make_app, 0.98)
    assert [a["tool"] for a in j["activity"]] == ["evaluate_labeled_data"]
    assert j["activity"][0]["result"]["accuracy_criterion"]["meets_min_accuracy"] is False
    assert "No prediction" in j["reply"]


def test_conditional_does_not_predict_when_evaluation_fails(make_app):
    app, c, llm = make_app()
    fid = upload(c, "samples/unlabeled_batch.csv").get_json()["file_id"]  # no labels -> eval fails
    llm.steps = [
        lambda m: response(tool_block("evaluate_labeled_data", {"file_id": fid, "min_accuracy": 0.9})),
        _conditional_policy(),
    ]
    j = chat(c, "evaluate then predict").get_json()
    assert [a["tool"] for a in j["activity"]] == ["evaluate_labeled_data"]
    assert j["activity"][0]["status"] == "error" and "labeled CSV" in j["activity"][0]["error"]
    assert j["tool_errors"] == 1 and j["warnings"]


def test_false_negative_follow_up_uses_stored_result(make_app):
    app, c, llm = make_app()
    fid = upload(c, "samples/labeled_demo_mixed.csv").get_json()["file_id"]
    llm.steps = [
        lambda m: response(tool_block("evaluate_labeled_data", {"file_id": fid})),
        lambda m: response(text_block("evaluated")),
    ]
    j1 = chat(c, "evaluate").get_json()
    rid = j1["activity"][0]["result"]["result_id"]
    fn_expected = j1["activity"][0]["result"]["confusion_matrix"]["false_negatives"]
    llm.steps = [
        lambda m: response(tool_block("query_results", {"result_id": rid, "query": "false_negatives"})),
        lambda m: response(text_block(f"{last_tool_results(m)[0]['count']} false negatives")),
    ]
    j2 = chat(c, "how many false negatives?").get_json()
    assert j2["reply"] == f"{fn_expected} false negatives"
    # the follow-up request carries the earlier turn in its history
    assert any("evaluate" in json.dumps(msg) for msg in llm.calls[-2]["messages"][:2])


def test_tool_failure_is_reported_not_hidden(make_app):
    steps = [
        lambda m: response(tool_block("predict_single_record", {"example": "example_malware"})),
        lambda m: response(text_block("The prediction tool failed." if last_tool_results(m)[0]["is_error"] else "malware")),
    ]
    app, c, llm = make_app(steps, FAULT_INJECTION=True)
    j = chat(c, "classify", simulate_tool_failure=True).get_json()
    a = j["activity"][0]
    assert a["status"] == "error" and "simulated" in a["error"] and "result" not in a
    assert j["tool_errors"] == 1 and "failed" in j["warnings"][0]
    assert last_tool_results(llm.calls[1]["messages"])[0]["is_error"] is True


def test_fault_injection_disabled_by_default(make_app):
    steps = [
        lambda m: response(tool_block("predict_single_record", {"example": "example_malware"})),
        lambda m: response(text_block("ok")),
    ]
    _, c, _ = make_app(steps)
    j = chat(c, "classify", simulate_tool_failure=True).get_json()
    assert j["activity"][0]["status"] == "ok"


def test_invalid_arguments_and_unknown_tool(make_app):
    steps = [
        lambda m: response(
            tool_block("predict_single_record", {}, "a"),
            tool_block("predict_batch", {"file_id": "file_nope"}, "b"),
            tool_block("run_shell", {"cmd": "ls"}, "c"),
        ),
        lambda m: response(text_block("errors")),
    ]
    _, c, llm = make_app(steps)
    j = chat(c, "do things").get_json()
    errs = [a["error"] for a in j["activity"]]
    assert all(a["status"] == "error" for a in j["activity"])
    assert "exactly one" in errs[0] and "unknown file_id" in errs[1] and "unknown tool" in errs[2]
    assert j["tool_errors"] == 3


def test_malformed_record_rejected(make_app):
    rec = json.load(open("samples/single_record.json"))
    rec.pop("Entropy")
    steps = [
        lambda m: response(tool_block("predict_single_record", {"record": rec})),
        lambda m: response(text_block("invalid")),
    ]
    _, c, _ = make_app(steps)
    a = chat(c, "classify this").get_json()["activity"][0]
    assert a["status"] == "error" and "Entropy" in a["error"]


def test_llm_service_failure(make_app):
    _, c, llm = make_app([TimeoutError("timed out")])
    r = chat(c, "classify")
    assert r.status_code == 502
    assert "failed or timed out" in r.get_json()["error"]


def test_tool_call_limit(make_app):
    loop = lambda m: response(tool_block("predict_single_record", {"example": "example_malware"}))
    _, c, _ = make_app([loop] * 20)
    j = chat(c, "loop forever").get_json()
    assert sum(a["status"] == "ok" for a in j["activity"]) == 6
    assert "limit" in json.dumps(j["activity"])


def test_missing_api_key_gives_clear_error(tmp_path, monkeypatch):
    from app import create_app

    app = create_app(llm_client=None, store_dir=str(tmp_path))
    r = app.test_client().post("/api/chat", json={"message": "hi"})
    assert r.status_code == 502 and "ANTHROPIC_API_KEY" in r.get_json()["error"]
