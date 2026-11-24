"""Per-session, server-side storage for uploads, tool results and chat history.

Each browser session gets a random id (kept in a signed Flask cookie). Files and
results live under that id only, so sessions cannot see each other's data.
Everything is in memory / local disk and expires after SESSION_TTL_SECONDS —
adequate for a single free-tier instance (documented limitation).
"""

from __future__ import annotations

import secrets
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

SESSION_TTL_SECONDS = 2 * 60 * 60
MAX_FILES_PER_SESSION = 10
MAX_RESULTS_PER_SESSION = 30


@dataclass
class UploadedFile:
    file_id: str
    name: str
    path: Path
    kind: str  # "csv" | "json"
    n_rows: int
    has_label: bool


@dataclass
class Session:
    sid: str
    root: Path
    files: dict[str, UploadedFile] = field(default_factory=dict)
    results: dict[str, dict] = field(default_factory=dict)
    history: list[list[dict]] = field(default_factory=list)  # one list of API messages per turn
    last_seen: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def new_id(self, prefix: str) -> str:
        return f"{prefix}_{secrets.token_hex(4)}"

    def add_file(self, f: UploadedFile):
        if len(self.files) >= MAX_FILES_PER_SESSION:
            oldest = next(iter(self.files))
            self.files.pop(oldest).path.unlink(missing_ok=True)
        self.files[f.file_id] = f

    def add_result(self, kind: str, data: dict) -> str:
        rid = self.new_id(kind)
        if len(self.results) >= MAX_RESULTS_PER_SESSION:
            self.results.pop(next(iter(self.results)))
        self.results[rid] = {"kind": kind, "created": time.time(), **data}
        return rid


class SessionStore:
    def __init__(self, base_dir: str | None = None):
        self.base = Path(base_dir or tempfile.mkdtemp(prefix="malware-agent-"))
        self.base.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def get(self, sid: str | None) -> Session:
        with self._lock:
            self._expire()
            if sid and sid in self._sessions:
                s = self._sessions[sid]
            else:
                sid = secrets.token_urlsafe(16)
                root = self.base / sid
                root.mkdir(parents=True, exist_ok=True)
                s = Session(sid=sid, root=root)
                self._sessions[sid] = s
            s.last_seen = time.time()
            return s

    def reset(self, sid: str):
        with self._lock:
            s = self._sessions.pop(sid, None)
        if s:
            shutil.rmtree(s.root, ignore_errors=True)

    def _expire(self):
        now = time.time()
        for sid in [k for k, v in self._sessions.items() if now - v.last_seen > SESSION_TTL_SECONDS]:
            shutil.rmtree(self._sessions.pop(sid).root, ignore_errors=True)

    def __len__(self):
        return len(self._sessions)
