"""Flask application: chat UI, uploads, direct tool endpoints and /health."""

from __future__ import annotations

import io
import json
import logging
import os
import secrets
import time
from collections import defaultdict, deque

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file, session as cookie

from malware_ml.inference import get_model
from malware_ml.schema import LABEL_COL, MODEL_FEATURES, NULLABLE_FEATURES

from . import agent as agent_mod
from .store import SessionStore, UploadedFile
from .tools import EXAMPLES, ToolError, evaluate_labeled_data, predict_batch, predict_single_record

MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "5"))
MAX_UPLOAD_ROWS = int(os.environ.get("MAX_UPLOAD_ROWS", "20000"))
CHAT_RATE_LIMIT = int(os.environ.get("CHAT_RATE_LIMIT", "20"))  # requests per window per session
RATE_WINDOW_SECONDS = 600

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("app")


def create_app(llm_client=None, store_dir: str | None = None) -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
    app.config["MAX_CONTENT_LENGTH"] = int(MAX_UPLOAD_MB * 1024 * 1024)
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.config["SESSION_COOKIE_SECURE"] = os.environ.get("SESSION_COOKIE_SECURE", "false").lower() == "true"
    app.config["LLM_CLIENT"] = llm_client  # injected in tests; None = real Anthropic client
    app.config["FAULT_INJECTION"] = os.environ.get("ENABLE_FAULT_INJECTION", "false").lower() == "true"

    store = SessionStore(store_dir)
    rate: dict[str, deque] = defaultdict(deque)
    model = get_model()  # load at startup so a broken artifact fails fast
    log.info("loaded model %s v%s threshold=%s", model.name, model.version, model.threshold)

    def current_session():
        s = store.get(cookie.get("sid"))
        cookie["sid"] = s.sid
        return s

    def err(msg, code=400, **extra):
        return jsonify({"status": "error", "error": msg, **extra}), code

    # ------------------------------------------------------------------ pages
    @app.get("/")
    def index():
        current_session()
        return render_template(
            "index.html",
            model=model.info(),
            fault_injection=app.config["FAULT_INJECTION"],
            max_mb=MAX_UPLOAD_MB,
        )

    @app.get("/health")
    def health():
        return jsonify({
            "status": "ok",
            "model_loaded": True,
            "model_name": model.name,
            "model_version": model.version,
            "commit": os.environ.get("RENDER_GIT_COMMIT") or os.environ.get("GIT_COMMIT", "unknown"),
            "llm_configured": bool(os.environ.get("ANTHROPIC_API_KEY")) or app.config["LLM_CLIENT"] is not None,
            "llm_model": agent_mod.DEFAULT_MODEL,
            "active_sessions": len(store),
        })

    @app.get("/api/schema")
    def api_schema():
        md = model.metadata
        return jsonify({"model": model.info(), "feature_schema": md["feature_schema"], "preprocessing": md["preprocessing"]})

    @app.get("/api/examples/<name>")
    def api_example(name):
        if name not in EXAMPLES:
            return err(f"unknown example; choose from {list(EXAMPLES)}", 404)
        return jsonify(json.loads(EXAMPLES[name].read_text()))

    SAMPLES_DIR = EXAMPLES["example_malware"].parent
    SAMPLE_FILES = {p.name for p in SAMPLES_DIR.glob("*") if p.suffix in (".csv", ".json")}

    @app.get("/samples/<name>")
    def sample_file(name):
        if name not in SAMPLE_FILES:
            return err("unknown sample file", 404)
        return send_file(SAMPLES_DIR / name, as_attachment=True, download_name=name)

    # ------------------------------------------------------------------ uploads
    @app.post("/api/upload")
    def api_upload():
        s = current_session()
        f = request.files.get("file")
        if f is None or not f.filename:
            return err("no file uploaded (form field 'file')")
        name = os.path.basename(f.filename)[:100]
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if ext not in ("csv", "json"):
            return err("only .csv and .json files are accepted")
        raw = f.read()
        if not raw:
            return err("the file is empty")
        file_id = s.new_id("file")
        path = s.root / f"{file_id}.{ext}"
        if ext == "csv":
            try:
                df = pd.read_csv(io.BytesIO(raw), low_memory=False)
            except Exception as e:
                return err(f"could not parse CSV: {str(e)[:200]}")
            if len(df) == 0:
                return err("the CSV has a header but no data rows")
            if len(df) > MAX_UPLOAD_ROWS:
                return err(f"too many rows ({len(df)}); the limit is {MAX_UPLOAD_ROWS}")
            missing = [c for c in MODEL_FEATURES if c not in df.columns and c not in NULLABLE_FEATURES]
            if missing:
                return err(f"CSV is missing required feature column(s): {', '.join(missing)}", missing_columns=missing)
            n_rows, has_label = len(df), LABEL_COL in df.columns
        else:
            try:
                obj = json.loads(raw)
            except Exception:
                return err("invalid JSON")
            if isinstance(obj, list) and len(obj) == 1:
                obj = obj[0]
            if not isinstance(obj, dict):
                return err("JSON upload must be a single feature record object")
            n_rows, has_label = 1, LABEL_COL in obj
        path.write_bytes(raw)
        uf = UploadedFile(file_id=file_id, name=name, path=path, kind=ext, n_rows=n_rows, has_label=has_label)
        s.add_file(uf)
        log.info("upload session=%s… file=%s rows=%d", s.sid[:6], file_id, n_rows)
        return jsonify({"status": "ok", "file_id": file_id, "name": name, "kind": ext, "rows": n_rows, "has_label": has_label})

    @app.get("/api/files")
    def api_files():
        s = current_session()
        return jsonify([{"file_id": f.file_id, "name": f.name, "kind": f.kind, "rows": f.n_rows, "has_label": f.has_label} for f in s.files.values()])

    @app.errorhandler(413)
    def too_large(_e):
        return err(f"file too large (limit {MAX_UPLOAD_MB:g} MB)", 413)

    # ------------------------------------------------------------------ direct tool endpoints
    def _direct(fn, args):
        s = current_session()
        try:
            payload, _ = fn(s, args)
            return jsonify({"status": "ok", **payload})
        except ToolError as e:
            return err(str(e), 422)

    @app.post("/api/predict")
    def api_predict():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return err("send a JSON object: {\"record\": {...}} or {\"example\": \"example_malware\"} or {\"file_id\": ...}")
        args = body if any(k in body for k in ("record", "example", "file_id")) else {"record": body}
        return _direct(predict_single_record, args)

    @app.post("/api/batch")
    def api_batch():
        return _direct(predict_batch, request.get_json(silent=True) or {})

    @app.post("/api/evaluate")
    def api_evaluate():
        return _direct(evaluate_labeled_data, request.get_json(silent=True) or {})

    @app.get("/api/results/<rid>/download")
    def api_download(rid):
        s = current_session()
        res = s.results.get(rid)
        if res is None or res["kind"] not in ("batch", "eval"):
            return err("result not found in this session", 404)
        data = res["result"]
        rows = data["predictions"] if res["kind"] == "batch" else data["_row_results"]
        out = pd.DataFrame(rows)
        if data.get("invalid_rows"):
            inv = pd.DataFrame([{"row_id": r["row_id"], "status": "invalid", "errors": "; ".join(r["errors"])} for r in data["invalid_rows"]])
            out = pd.concat([out.assign(status="scored"), inv], ignore_index=True)
            out["_n"] = out["row_id"].str.replace("row_", "").astype(int)
            out = out.sort_values("_n").drop(columns="_n")
            for c in ("predicted_label", "true_label"):
                if c in out:
                    out[c] = out[c].astype("Int64")
        buf = io.BytesIO(out.to_csv(index=False).encode())
        return send_file(buf, mimetype="text/csv", as_attachment=True, download_name=f"{rid}_{res.get('file', 'results')}")

    # ------------------------------------------------------------------ agent chat
    @app.post("/api/chat")
    def api_chat():
        s = current_session()
        body = request.get_json(silent=True) or {}
        q = rate[s.sid]
        now = time.time()
        while q and now - q[0] > RATE_WINDOW_SECONDS:
            q.popleft()
        if len(q) >= CHAT_RATE_LIMIT:
            return err(f"rate limit: at most {CHAT_RATE_LIMIT} agent requests per {RATE_WINDOW_SECONDS // 60} minutes", 429)
        q.append(now)
        simulate = bool(body.get("simulate_tool_failure")) and app.config["FAULT_INJECTION"]
        if not s.lock.acquire(blocking=False):
            return err("the agent is still working on your previous request", 409)
        try:
            out = agent_mod.run_turn(s, body.get("message", ""), client=app.config["LLM_CLIENT"], simulate_tool_failure=simulate)
            log.info("chat session=%s… tools=%s elapsed=%.1fs", s.sid[:6], [a["tool"] + ":" + a["status"] for a in out["activity"]], out["elapsed_seconds"])
            return jsonify({"status": "ok", **out})
        except agent_mod.AgentError as e:
            log.warning("chat error session=%s…: %s", s.sid[:6], e)
            return err(str(e), 502)
        finally:
            s.lock.release()

    @app.post("/api/reset")
    def api_reset():
        sid = cookie.pop("sid", None)
        if sid:
            store.reset(sid)
        return jsonify({"status": "ok"})

    return app
