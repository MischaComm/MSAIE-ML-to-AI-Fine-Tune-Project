"""Run the required agent-evaluation scenarios against a running app that uses
the REAL configured AI model (not the CI mock), and record the evidence.

    python scripts/run_agent_eval.py --url https://<your-app>.onrender.com
    python scripts/run_agent_eval.py --url http://localhost:5000

Writes reports/agent_eval_results.json and reports/agent_eval_table.md.
The fault-injection scenario needs ENABLE_FAULT_INJECTION=true on the server.
Automatic checks are heuristics; review each reply before finalising
agent-evaluation.md.
"""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import re
import sys
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"


class Client:
    def __init__(self, base):
        self.base = base.rstrip("/")
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))

    def _req(self, path, data=None, headers=None, timeout=180):
        req = urllib.request.Request(self.base + path, data=data, headers=headers or {})
        try:
            with self.opener.open(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            return json.loads(e.read() or b"{}")

    def get(self, path):
        return self._req(path)

    def chat(self, message, simulate_tool_failure=False):
        body = json.dumps({"message": message, "simulate_tool_failure": simulate_tool_failure}).encode()
        return self._req("/api/chat", body, {"Content-Type": "application/json"})

    def upload(self, path: Path):
        boundary = uuid.uuid4().hex
        payload = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
            f"Content-Type: text/csv\r\n\r\n"
        ).encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        return self._req("/api/upload", payload, {"Content-Type": f"multipart/form-data; boundary={boundary}"})


def tools_called(resp):
    return [a["tool"] for a in resp.get("activity", [])]


def pct(x):
    return f"{x * 100:.2f}".rstrip("0").rstrip(".")


def mentions_number(reply: str, value) -> bool:
    """True if the reply states the value (as a fraction, or as a percentage)."""
    if value is None:
        return False
    text = reply.replace(",", "")
    candidates = {str(value), f"{value:.4f}".rstrip("0"), f"{value:.3f}".rstrip("0")} if isinstance(value, float) else {str(value)}
    if isinstance(value, float) and value <= 1:
        candidates |= {pct(value), f"{value * 100:.1f}", f"{value * 100:.0f}%"}
    return any(re.search(rf"(?<![\d.]){re.escape(c)}(?![\d])", text) for c in candidates if c)


def result_of(resp, tool):
    for a in resp.get("activity", []):
        if a["tool"] == tool and a["status"] == "ok":
            return a["result"]
    return None


def scenario(name, request, expected, resp, passed, checks):
    return {
        "scenario": name,
        "request": request,
        "expected": expected,
        "tools_called": [
            {"tool": a["tool"], "arguments": a["arguments"], "status": a["status"], **({"error": a["error"]} if a["status"] != "ok" else {"result": a.get("result")})}
            for a in resp.get("activity", [])
        ],
        "reply": resp.get("reply") or resp.get("error"),
        "warnings": resp.get("warnings", []),
        "numerical_checks": checks,
        "pass": bool(passed),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    args = ap.parse_args()
    health = Client(args.url).get("/health")
    print("health:", health)
    if health.get("status") != "ok":
        sys.exit("app is not healthy")
    results = []

    # 1. single prediction
    c = Client(args.url)
    q = "Classify the pre-filled example malware record."
    r = c.chat(q)
    res = result_of(r, "predict_single_record") or {}
    checks = {"reply states tool's class": res.get("predicted_class", "~") in (r.get("reply") or "").lower()}
    results.append(scenario("1. Single-record prediction", q, "Calls predict_single_record (example_malware) and reports the model's class, score and version without calling it a calibrated probability.",
                            r, tools_called(r) == ["predict_single_record"] and all(checks.values()), checks))

    # 2. batch prediction
    c = Client(args.url)
    c.upload(SAMPLES / "unlabeled_batch.csv")
    q = "Classify every row in unlabeled_batch.csv and give me the malware and goodware counts."
    r = c.chat(q)
    res = result_of(r, "predict_batch") or {}
    s = res.get("summary", {})
    checks = {"malware count stated": mentions_number(r.get("reply", ""), s.get("malware")),
              "goodware count stated": mentions_number(r.get("reply", ""), s.get("goodware")),
              "download link returned": bool(r.get("links"))}
    results.append(scenario("2. Batch prediction", q, "Calls predict_batch with the uploaded file id; reports counts matching the tool summary and links per-row results.",
                            r, tools_called(r) == ["predict_batch"] and all(checks.values()), checks))

    # 3. conditional - permitted
    c = Client(args.url)
    c.upload(SAMPLES / "labeled_demo_mixed.csv")
    q = "Evaluate labeled_demo_mixed.csv. If the accuracy is at least 90%, classify the pre-filled example goodware record; otherwise don't make the prediction."
    r = c.chat(q)
    ev = result_of(r, "evaluate_labeled_data") or {}
    m = ev.get("metrics", {})
    checks = {"accuracy stated": mentions_number(r.get("reply", ""), m.get("accuracy")),
              "AUC stated": mentions_number(r.get("reply", ""), m.get("auc")),
              "criterion met per tool": (ev.get("accuracy_criterion") or {}).get("meets_min_accuracy") is True}
    results.append(scenario("3a. Labeled evaluation + conditional prediction (permitted)", q,
                            "evaluate_labeled_data (min_accuracy 0.9) -> accuracy 0.94 >= 0.90 -> predict_single_record is called and reported.",
                            r, tools_called(r) == ["evaluate_labeled_data", "predict_single_record"] and all(checks.values()), checks))

    # 4. conditional - withheld, then 5. false-negative follow-up in the same session
    c = Client(args.url)
    c.upload(SAMPLES / "labeled_demo_mixed.csv")
    q = "Evaluate labeled_demo_mixed.csv. If the accuracy is at least 98%, classify the pre-filled example goodware record; otherwise don't make the prediction."
    r = c.chat(q)
    ev = result_of(r, "evaluate_labeled_data") or {}
    fn = (ev.get("confusion_matrix") or {}).get("false_negatives")
    checks = {"criterion not met per tool": (ev.get("accuracy_criterion") or {}).get("meets_min_accuracy") is False,
              "explains withheld": bool(re.search(r"not (been )?made|no prediction|withh|did not (make|classify)|didn't", (r.get("reply") or "").lower()))}
    results.append(scenario("3b. Labeled evaluation + conditional prediction (withheld)", q,
                            "evaluate_labeled_data (min_accuracy 0.98) -> accuracy 0.94 < 0.98 -> predict_single_record NOT called; agent explains why.",
                            r, tools_called(r) == ["evaluate_labeled_data"] and all(checks.values()), checks))
    q = "How many false negatives were there in that evaluation, and which rows were they?"
    r = c.chat(q)
    checks = {"false-negative count matches confusion matrix": mentions_number(r.get("reply", ""), fn)}
    results.append(scenario("4. False-negative follow-up", q, f"Answers from the stored evaluation result (count = {fn}), optionally calling query_results for row ids; defines FN as true malware predicted goodware.",
                            r, all(checks.values()) and "predict_single_record" not in tools_called(r), checks))

    # 6. invalid input
    c = Client(args.url)
    c.upload(SAMPLES / "batch_with_invalid_rows.csv")
    q = "Classify batch_with_invalid_rows.csv."
    r = c.chat(q)
    res = result_of(r, "predict_batch") or {}
    reply = r.get("reply") or ""
    checks = {"invalid count stated": mentions_number(reply, res.get("n_invalid")), "invalid row ids named": "row_4" in reply and "row_7" in reply}
    results.append(scenario("5. Invalid input", q, "predict_batch scores valid rows and reports row_4 (non-numeric Entropy) and row_7 (missing NumberOfSections) as invalid, not dropped.",
                            r, tools_called(r) == ["predict_batch"] and all(checks.values()), checks))

    # 7. missing labels
    c = Client(args.url)
    c.upload(SAMPLES / "unlabeled_batch.csv")
    q = "Evaluate the model's accuracy on unlabeled_batch.csv."
    r = c.chat(q)
    reply = (r.get("reply") or "").lower()
    ok_tools = all(t in ("evaluate_labeled_data",) for t in tools_called(r))
    checks = {"asks for labeled file": "label" in reply, "no metrics reported": result_of(r, "evaluate_labeled_data") is None}
    results.append(scenario("6. Missing labels", q, "Does not report metrics; explains the file has no Label column and asks for a labeled CSV (tool error or clarification).",
                            r, ok_tools and all(checks.values()), checks))

    # 8. single-class evaluation
    c = Client(args.url)
    c.upload(SAMPLES / "labeled_single_class_malware.csv")
    q = "Evaluate labeled_single_class_malware.csv."
    r = c.chat(q)
    ev = result_of(r, "evaluate_labeled_data") or {}
    checks = {"AUC reported undefined by tool": (ev.get("metrics") or {}).get("auc") is None,
              "reply explains AUC undefined": "undefined" in (r.get("reply") or "").lower() or "not defined" in (r.get("reply") or "").lower(),
              "accuracy stated": mentions_number(r.get("reply", ""), (ev.get("metrics") or {}).get("accuracy"))}
    results.append(scenario("7. Single-class evaluation", q, "evaluate_labeled_data returns auc=null; agent explains AUC is undefined with one class and reports accuracy / confusion matrix.",
                            r, tools_called(r) == ["evaluate_labeled_data"] and all(checks.values()), checks))

    # 9. tool / service failure (controlled fault)
    c = Client(args.url)
    q = "Classify the pre-filled example malware record."
    r = c.chat(q, simulate_tool_failure=True)
    reply = (r.get("reply") or "").lower()
    checks = {"tool status error": any(a["status"] == "error" for a in r.get("activity", [])),
              "reply admits failure": bool(re.search(r"fail|error|unable|could not|couldn't", reply)),
              "no class claimed": not re.search(r"\b(is|as|predicted)\s+(\*\*)?malware\b", reply)}
    results.append(scenario("8. Tool / service failure (controlled fault)", q + "  [simulate_tool_failure=true]",
                            "The tool returns an error; the agent says the step failed and does not present a prediction.",
                            r, all(checks.values()), checks))

    REP = ROOT / "reports"
    REP.mkdir(exist_ok=True)
    meta = {"evaluated_at": datetime.now(timezone.utc).isoformat(), "url": args.url, "llm_model": health.get("llm_model"),
            "app_commit": health.get("commit"), "model": f"{health.get('model_name')} v{health.get('model_version')}"}
    n_pass = sum(x["pass"] for x in results)
    out = {**meta, "pass_rate": f"{n_pass}/{len(results)}", "scenarios": results}
    (REP / "agent_eval_results.json").write_text(json.dumps(out, indent=2))

    lines = [f"Evaluated {meta['evaluated_at'][:10]} against {args.url} · AI model `{meta['llm_model']}` · ML model {meta['model']} · commit `{meta['app_commit']}`", "",
             f"**Automatic pass rate: {n_pass}/{len(results)}**", "",
             "| # | Scenario | Tools actually called | Numerical checks | Pass |", "|---|---|---|---|---|"]
    for x in results:
        tools = " → ".join(f"`{t['tool']}`({t['status']})" for t in x["tools_called"]) or "none"
        chk = "; ".join(f"{k}: {'✓' if v else '✗'}" for k, v in x["numerical_checks"].items())
        lines.append(f"| {x['scenario'].split('.')[0]} | {x['scenario'].split('. ', 1)[1]} | {tools} | {chk} | {'PASS' if x['pass'] else 'FAIL'} |")
    (REP / "agent_eval_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nFull evidence (requests, arguments, replies): reports/agent_eval_results.json")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"done in {time.time() - t0:.0f}s")
