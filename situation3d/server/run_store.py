"""Run store for the TW2 dual-mode workflow.

A *run* is one immutable simulation output produced by
``scripts/run_situation_case.py``.  The store owns the directory layout,
atomic writes, checksums, role bookkeeping (latest / best / pinned) and the
three-slot retention policy confirmed by the user.

Layout::

    <root>/<case_id>/
        .lock
        state.json                 # roles + status index (rebuildable)
        audit.jsonl                # retention / selection audit trail
        <run_id>/
            run.json               # sim.run.v1 metadata
            scenario.json          # initial scenario snapshot
            trajectory.json        # situation3d.trajectory.v2
            nodes.json             # key situation nodes
            summary.json
            checksums.sha256
            run.log

Roles (at most one run each):
    latest  - the most recently completed run
    best    - user-selected run, the only promotable one
    pinned  - one extra run the user explicitly keeps

A run may hold several roles at once; distinct runs with roles never exceed
``keep`` (default 3).  Failed runs do not consume a slot; only the newest
failed run is kept for diagnosis.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import threading
import time
import uuid

try:  # Linux server; the fallback keeps Windows dev usable.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

SCHEMA_STATE = "sim.runs.state.v1"
SCHEMA_RUN = "sim.run.v1"
ROLES = ("latest", "best", "pinned")


def _ordered_roles(roles):
    wanted = set(roles or [])
    return [r for r in ROLES if r in wanted]


DEFAULT_KEEP = 3
DEFAULT_FAILED_KEEP = 1

CASE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
RUN_RE = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{8}$")
ARTIFACT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

_LOCKS = {}
_LOCKS_GUARD = threading.Lock()


class RunError(Exception):
    """Base error for run-store operations."""


class RunNotFound(RunError):
    pass


class RunConflict(RunError):
    def __init__(self, message, **detail):
        super(RunConflict, self).__init__(message)
        self.detail = detail


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_write_bytes(path, data):
    tmp = "%s.tmp.%d" % (path, os.getpid())
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _atomic_write_json(path, obj):
    data = json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True).encode("utf-8")
    _atomic_write_bytes(path, data)


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class RunStore(object):
    """File-backed, lock-protected run library with role retention."""

    def __init__(self, root, keep=DEFAULT_KEEP, failed_keep=DEFAULT_FAILED_KEEP):
        self.root = os.path.abspath(root)
        self.keep = int(keep)
        self.failed_keep = int(failed_keep)
        if self.keep < 1:
            raise ValueError("keep must be >= 1")
        if self.failed_keep < 0:
            raise ValueError("failed_keep must be >= 0")
        os.makedirs(self.root, exist_ok=True)

    # -- paths / validation -------------------------------------------------

    @staticmethod
    def _validate_case(case_id):
        if not isinstance(case_id, str) or not CASE_RE.match(case_id) or ".." in case_id:
            raise RunError("invalid case_id: %r" % (case_id,))

    @staticmethod
    def _validate_run(run_id):
        if not isinstance(run_id, str) or not RUN_RE.match(run_id):
            raise RunError("invalid run_id: %r" % (run_id,))

    def _case_dir(self, case_id):
        self._validate_case(case_id)
        return os.path.join(self.root, case_id)

    def _run_dir(self, case_id, run_id):
        self._validate_run(run_id)
        return os.path.join(self._case_dir(case_id), run_id)

    def _thread_lock(self, key):
        with _LOCKS_GUARD:
            lock = _LOCKS.get(key)
            if lock is None:
                lock = threading.Lock()
                _LOCKS[key] = lock
            return lock

    @contextlib.contextmanager
    def _lock(self, case_dir):
        os.makedirs(case_dir, exist_ok=True)
        path = os.path.join(case_dir, ".lock")
        fh = open(path, "a+")
        if fcntl is None:
            lock = self._thread_lock(path)
            with lock:
                yield
            fh.close()
            return
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            fh.close()

    # -- state --------------------------------------------------------------

    def _state_path(self, case_dir):
        return os.path.join(case_dir, "state.json")

    def _load_state(self, case_dir):
        path = self._state_path(case_dir)
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    state = json.load(fh)
                if state.get("schema") == SCHEMA_STATE and isinstance(state.get("runs"), dict):
                    return self._reconcile_state(case_dir, state)
            except (OSError, ValueError):
                pass
        state = self._rebuild_state(case_dir)
        _atomic_write_json(path, state)
        return state

    def _reconcile_state(self, case_dir, state):
        """Keep the index honest when a crash splits run.json and state.json."""
        changed = False
        best = None
        for rid, ent in list(state.get("runs", {}).items()):
            try:
                run = self._read_run(case_dir, rid)
            except RunNotFound:
                state["runs"].pop(rid, None)
                changed = True
                continue
            fresh = self._entry(run)
            if fresh != ent:
                state["runs"][rid] = fresh
                changed = True
            if "best" in fresh.get("roles", []):
                best = rid
        if state.get("best") != best:
            state["best"] = best
            changed = True
        if changed:
            self._save_state(case_dir, state)
        return state

    def _rebuild_state(self, case_dir):
        case_id = os.path.basename(case_dir)
        runs = {}
        best = None
        try:
            names = sorted(os.listdir(case_dir))
        except OSError:
            names = []
        for name in names:
            if not RUN_RE.match(name):
                continue
            run_p = os.path.join(case_dir, name, "run.json")
            if not os.path.isfile(run_p):
                continue
            try:
                with open(run_p, "r", encoding="utf-8") as fh:
                    run = json.load(fh)
            except (OSError, ValueError):
                continue
            roles = _ordered_roles(run.get("roles"))
            runs[name] = {
                "status": run.get("status"),
                "created_at": run.get("created_at"),
                "created_ns": run.get("created_ns"),
                "finished_at": run.get("finished_at"),
                "finished_ns": run.get("finished_ns"),
                "roles": roles,
                "size_bytes": run.get("size_bytes", 0),
                "metrics": run.get("metrics") or {},
                "validation": run.get("validation") or {},
                "seed": run.get("seed"),
                "progress": run.get("progress") or {},
            }
            if "best" in roles:
                best = name
        return {"schema": SCHEMA_STATE, "case_id": case_id,
                "updated_at": utc_now(), "best": best, "runs": runs}

    @staticmethod
    def _save_state(case_dir, state):
        state["updated_at"] = utc_now()
        _atomic_write_json(os.path.join(case_dir, "state.json"), state)

    def _audit(self, case_dir, action, run_id=None, **detail):
        rec = {"at": utc_now(), "action": action, "run_id": run_id}
        rec.update(detail)
        path = os.path.join(case_dir, "audit.jsonl")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            fh.flush()

    def _entry(self, run):
        return {
            "status": run.get("status"),
            "created_at": run.get("created_at"),
            "created_ns": run.get("created_ns"),
            "finished_at": run.get("finished_at"),
            "finished_ns": run.get("finished_ns"),
            "roles": _ordered_roles(run.get("roles")),
            "size_bytes": int(run.get("size_bytes") or 0),
            "metrics": run.get("metrics") or {},
            "validation": run.get("validation") or {},
            "seed": run.get("seed"),
            "progress": run.get("progress") or {},
            "scenario_date": (run.get("config") or {}).get("scenario_date"),
            "resources": run.get("resources") or {},
            "resource_gaps": run.get("resource_gaps") or [],
        }

    def _read_run(self, case_dir, run_id):
        path = os.path.join(case_dir, run_id, "run.json")
        if not os.path.isfile(path):
            raise RunNotFound("run not found: %s" % run_id)
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _write_run(self, case_dir, run_id, run):
        run["updated_at"] = utc_now()
        _atomic_write_json(os.path.join(case_dir, run_id, "run.json"), run)

    def _remove_role(self, case_dir, state, role, except_run=None):
        for rid, ent in state.get("runs", {}).items():
            if rid == except_run:
                continue
            roles = ent.get("roles") or []
            if role in roles:
                new_roles = [r for r in roles if r != role]
                ent["roles"] = new_roles
                try:
                    run = self._read_run(case_dir, rid)
                    run["roles"] = new_roles
                    self._write_run(case_dir, rid, run)
                except RunNotFound:
                    pass

    def _delete_run_locked(self, case_dir, state, run_id, reason):
        run_dir = os.path.join(case_dir, run_id)
        if os.path.isdir(run_dir):
            shutil.rmtree(run_dir, ignore_errors=True)
        state.get("runs", {}).pop(run_id, None)
        if state.get("best") == run_id:
            state["best"] = None
        self._audit(case_dir, "delete", run_id, reason=reason)

    def _drop_unroled_locked(self, case_dir, state):
        for rid, ent in list(state.get("runs", {}).items()):
            if ent.get("status") in ("queued", "running",
                                     "waiting_resources", "failed"):
                continue
            if ent.get("roles"):
                continue
            self._delete_run_locked(case_dir, state, rid, "no retained role")

    def _prune_locked(self, case_dir, state):
        # Stale entries whose directory disappeared.
        for rid in list(state.get("runs", {})):
            if not os.path.isdir(os.path.join(case_dir, rid)):
                state["runs"].pop(rid, None)
        # Completed runs that lost every role.
        self._drop_unroled_locked(case_dir, state)
        # Failed runs: keep only the newest N, not part of the 3 slots.
        failed = [(rid, ent) for rid, ent in state.get("runs", {}).items()
                  if ent.get("status") == "failed"]
        failed.sort(key=lambda kv: (kv[1].get("finished_ns") or 0,
                                    kv[1].get("created_ns") or 0),
                    reverse=True)
        for rid, _ in failed[self.failed_keep:]:
            self._delete_run_locked(case_dir, state, rid, "failed run retention")
        # Defensive cap: roles are exclusive per slot, so this should not fire.
        retained = [(rid, ent) for rid, ent in state.get("runs", {}).items()
                    if ent.get("roles")]
        if len(retained) > self.keep:
            retained.sort(key=lambda kv: kv[1].get("created_at") or "")
            for rid, ent in retained:
                if len([1 for _r, e in state["runs"].items() if e.get("roles")]) <= self.keep:
                    break
                if "best" in (ent.get("roles") or []):
                    continue
                self._delete_run_locked(case_dir, state, rid, "retention cap")

    # -- public API ---------------------------------------------------------

    def create_run(self, case_id, seed=None, config=None, scenario_hash=None,
                   engine_commit=None, envpack_hash=None, run_id=None,
                   reject_active=False):
        case_dir = self._case_dir(case_id)
        with self._lock(case_dir):
            state = self._load_state(case_dir)
            if reject_active:
                for rid, ent in state.get("runs", {}).items():
                    if ent.get("status") in ("queued", "running",
                                             "waiting_resources"):
                        raise RunConflict("another run is already active",
                                          active=rid)
            if run_id is not None:
                self._validate_run(run_id)
                if os.path.exists(os.path.join(case_dir, run_id)):
                    raise RunConflict("run already exists: %s" % run_id)
            else:
                run_id = self._new_run_id(case_dir)
            os.makedirs(os.path.join(case_dir, run_id), exist_ok=True)
            run = {
                "schema": SCHEMA_RUN,
                "run_id": run_id,
                "case_id": case_id,
                "status": "queued",
                "created_at": utc_now(),
                "created_ns": time.time_ns(),
                "started_at": None,
                "finished_at": None,
                "finished_ns": None,
                "roles": [],
                "seed": seed,
                "config": config or {},
                "scenario_hash": scenario_hash,
                "engine_commit": engine_commit,
                "envpack_hash": envpack_hash,
                "metrics": {},
                "validation": {},
                "artifacts": {},
                "size_bytes": 0,
            }
            self._write_run(case_dir, run_id, run)
            state["runs"][run_id] = self._entry(run)
            self._save_state(case_dir, state)
            self._audit(case_dir, "create", run_id)
            return run_id

    def update_run(self, case_id, run_id, status=None, **fields):
        case_dir = self._case_dir(case_id)
        with self._lock(case_dir):
            state = self._load_state(case_dir)
            run = self._read_run(case_dir, run_id)
            old = run.get("status")
            if status and status != old:
                allowed = {
                    "queued": ("running", "waiting_resources", "failed",
                               "cancelled"),
                    "running": ("failed", "cancelled"),
                    "waiting_resources": ("queued", "failed", "cancelled"),
                }
                if status not in allowed.get(old, ()):
                    raise RunConflict("invalid transition %s -> %s" % (old, status))
                run["status"] = status
                if status == "running" and not run.get("started_at"):
                    run["started_at"] = utc_now()
            for key, value in fields.items():
                if key in ("run_id", "case_id", "roles", "schema"):
                    continue
                run[key] = value
            self._write_run(case_dir, run_id, run)
            state["runs"][run_id] = self._entry(run)
            self._save_state(case_dir, state)
            return run

    def complete_run(self, case_id, run_id, artifacts, metrics=None, validation=None):
        case_dir = self._case_dir(case_id)
        with self._lock(case_dir):
            state = self._load_state(case_dir)
            run = self._read_run(case_dir, run_id)
            if run.get("status") not in ("queued", "running"):
                raise RunConflict("run is already %s" % run.get("status"))
            run_dir = os.path.join(case_dir, run_id)
            hashes = {}
            for name, payload in (artifacts or {}).items():
                if not ARTIFACT_RE.match(name) or name in ("run.json", "state.json"):
                    raise RunError("invalid artifact name: %r" % (name,))
                target = os.path.join(run_dir, name)
                if isinstance(payload, bytes):
                    _atomic_write_bytes(target, payload)
                elif isinstance(payload, str):
                    if not os.path.isfile(payload):
                        raise RunError("artifact source missing: %s" % payload)
                    tmp = target + ".tmp.%d" % os.getpid()
                    shutil.copyfile(payload, tmp)
                    os.replace(tmp, target)
                else:
                    raise RunError("artifact %s must be bytes or path" % name)
                hashes[name] = _sha256_file(target)
            checks = "".join("%s  %s\n" % (hashes[n], n) for n in sorted(hashes))
            _atomic_write_bytes(os.path.join(run_dir, "checksums.sha256"),
                                checks.encode("utf-8"))
            run["status"] = "completed"
            run["finished_at"] = utc_now()
            run["finished_ns"] = time.time_ns()
            run["artifacts"] = hashes
            run["metrics"] = metrics or {}
            run["validation"] = validation or {}
            self._remove_role(case_dir, state, "latest", except_run=run_id)
            roles = _ordered_roles([r for r in (run.get("roles") or [])
                                    if r != "latest"] + ["latest"])
            run["roles"] = roles
            run["size_bytes"] = self._dir_size(run_dir)
            self._write_run(case_dir, run_id, run)
            state["runs"][run_id] = self._entry(run)
            state["best"] = run_id if "best" in roles else state.get("best")
            self._prune_locked(case_dir, state)
            self._save_state(case_dir, state)
            self._audit(case_dir, "complete", run_id,
                        size_bytes=run["size_bytes"])
            return run

    def fail_run(self, case_id, run_id, error, log_text=None):
        case_dir = self._case_dir(case_id)
        with self._lock(case_dir):
            state = self._load_state(case_dir)
            run = self._read_run(case_dir, run_id)
            if run.get("status") in ("completed", "cancelled"):
                raise RunConflict("run is already %s" % run.get("status"))
            run_dir = os.path.join(case_dir, run_id)
            if log_text is not None:
                _atomic_write_bytes(os.path.join(run_dir, "run.log"),
                                    str(log_text).encode("utf-8", "replace"))
            run["status"] = "failed"
            run["finished_at"] = utc_now()
            run["finished_ns"] = time.time_ns()
            run["error"] = str(error)
            run["roles"] = [r for r in (run.get("roles") or []) if r == "best" or r == "pinned"]
            self._write_run(case_dir, run_id, run)
            state["runs"][run_id] = self._entry(run)
            self._prune_locked(case_dir, state)
            self._save_state(case_dir, state)
            self._audit(case_dir, "fail", run_id, error=str(error))
            return run

    def list_runs(self, case_id, include_failed=True):
        case_dir = self._case_dir(case_id)
        with self._lock(case_dir):
            state = self._load_state(case_dir)
            rows = []
            for rid, ent in state.get("runs", {}).items():
                if not include_failed and ent.get("status") == "failed":
                    continue
                row = dict(ent)
                row["run_id"] = rid
                row["case_id"] = case_id
                rows.append(row)
        rows.sort(key=lambda r: (r.get("created_ns") or 0,
                                 r.get("created_at") or ""), reverse=True)
        return rows

    def get_run(self, case_id, run_id):
        case_dir = self._case_dir(case_id)
        with self._lock(case_dir):
            state = self._load_state(case_dir)
            run = self._read_run(case_dir, run_id)
            run["roles"] = list((state.get("runs", {}).get(run_id) or {}).get("roles") or [])
            return run

    def run_dir(self, case_id, run_id):
        """Public path to a run directory (after validating the run exists)."""
        self.get_run(case_id, run_id)
        return self._run_dir(case_id, run_id)

    def read_artifact(self, case_id, run_id, name):
        if not ARTIFACT_RE.match(name or "") or name in ("run.json", "checksums.sha256"):
            raise RunError("invalid artifact name: %r" % (name,))
        path = os.path.join(self._run_dir(case_id, run_id), name)
        if not os.path.isfile(path):
            raise RunNotFound("artifact not found: %s" % name)
        return path

    def set_best(self, case_id, run_id):
        case_dir = self._case_dir(case_id)
        with self._lock(case_dir):
            state = self._load_state(case_dir)
            run = self._read_run(case_dir, run_id)
            if run.get("status") != "completed":
                raise RunConflict("only completed runs can be selected",
                                  status=run.get("status"))
            if not (run.get("validation") or {}).get("passed", False):
                raise RunConflict("run failed validation; fix or choose another run")
            self._remove_role(case_dir, state, "best", except_run=run_id)
            roles = _ordered_roles([r for r in (run.get("roles") or [])
                                    if r != "best"] + ["best"])
            run["roles"] = roles
            self._write_run(case_dir, run_id, run)
            state["runs"][run_id] = self._entry(run)
            state["best"] = run_id
            self._drop_unroled_locked(case_dir, state)
            self._save_state(case_dir, state)
            self._audit(case_dir, "select_best", run_id)
            return run

    def pin(self, case_id, run_id, replace=None):
        case_dir = self._case_dir(case_id)
        with self._lock(case_dir):
            state = self._load_state(case_dir)
            run = self._read_run(case_dir, run_id)
            if run.get("status") != "completed":
                raise RunConflict("only completed runs can be pinned",
                                  status=run.get("status"))
            current = None
            for rid, ent in state.get("runs", {}).items():
                if "pinned" in (ent.get("roles") or []):
                    current = rid
                    break
            if current and current != run_id and replace != current:
                raise RunConflict("a run is already pinned; pass replace",
                                  pinned=current)
            if current and current != run_id:
                old = state["runs"][current]
                old["roles"] = [r for r in old.get("roles") or [] if r != "pinned"]
                try:
                    old_run = self._read_run(case_dir, current)
                    old_run["roles"] = old["roles"]
                    self._write_run(case_dir, current, old_run)
                except RunNotFound:
                    pass
                if not old["roles"]:
                    self._delete_run_locked(case_dir, state, current, "unpinned")
            roles = _ordered_roles([r for r in (run.get("roles") or [])
                                    if r != "pinned"] + ["pinned"])
            run["roles"] = roles
            self._write_run(case_dir, run_id, run)
            state["runs"][run_id] = self._entry(run)
            self._drop_unroled_locked(case_dir, state)
            self._save_state(case_dir, state)
            self._audit(case_dir, "pin", run_id, replaced=current)
            return run

    def unpin(self, case_id, run_id):
        case_dir = self._case_dir(case_id)
        with self._lock(case_dir):
            state = self._load_state(case_dir)
            run = self._read_run(case_dir, run_id)
            run["roles"] = [r for r in (run.get("roles") or []) if r != "pinned"]
            self._write_run(case_dir, run_id, run)
            state["runs"][run_id] = self._entry(run)
            self._drop_unroled_locked(case_dir, state)
            self._save_state(case_dir, state)
            self._audit(case_dir, "unpin", run_id)
            return run

    def delete_run(self, case_id, run_id):
        case_dir = self._case_dir(case_id)
        with self._lock(case_dir):
            state = self._load_state(case_dir)
            run = self._read_run(case_dir, run_id)
            if "best" in (run.get("roles") or []):
                raise RunConflict("best run cannot be deleted; select another first")
            if "pinned" in (run.get("roles") or []):
                raise RunConflict("pinned run must be unpinned first")
            if run.get("status") in ("queued", "running",
                                     "waiting_resources"):
                raise RunConflict("active run cannot be deleted",
                                  status=run.get("status"))
            self._delete_run_locked(case_dir, state, run_id, "user delete")
            self._save_state(case_dir, state)
            return True

    def audit_note(self, case_id, run_id, action, note=""):
        case_dir = self._case_dir(case_id)
        with self._lock(case_dir):
            self._audit(case_dir, str(action), run_id, note=str(note or ""))

    def verify(self, case_id, run_id):
        run_dir = self._run_dir(case_id, run_id)
        path = os.path.join(run_dir, "checksums.sha256")
        if not os.path.isfile(path):
            raise RunNotFound("checksums missing for %s" % run_id)
        bad = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line:
                    continue
                digest, _, name = line.partition("  ")
                full = os.path.join(run_dir, name)
                if not os.path.isfile(full) or _sha256_file(full) != digest:
                    bad.append(name)
        return {"ok": not bad, "mismatches": bad}

    # -- helpers ------------------------------------------------------------

    def _new_run_id(self, case_dir):
        for _ in range(20):
            rid = "%s-%s" % (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
                             uuid.uuid4().hex[:8])
            if not os.path.exists(os.path.join(case_dir, rid)):
                return rid
        raise RunError("could not allocate a run id")

    @staticmethod
    def _dir_size(path):
        total = 0
        for root, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    pass
        return total
