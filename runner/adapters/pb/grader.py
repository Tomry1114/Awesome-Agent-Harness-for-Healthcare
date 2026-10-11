"""native_pytest executor: run a PhysicianBench checkpoint (tests/test_outputs.py::func) the same way
upstream scripts/run_eval.py does — subprocess pytest with FHIR_BASE_URL + JOB_DIR env, cwd=repo.

Maps the outcome to (checkpoint_status, failure_mode):
  any judge call failed -> error  / judge_error        (whatever the assertion said)
  rc 0                  -> passed
  rc 1 + Assertion      -> failed / agent_failure
  rc 1 + conn err       -> error  / environment_error
  rc 1 + other exc      -> error  / verifier_error
  rc >=2                -> error  / verifier_error   (collection/usage/internal)

The judge row comes first. Upstream utils/eval_helpers.llm_judge catches a judge exception and returns FAIL, so
the assertion that follows fails exactly like an agent that wrote the wrong thing. The subprocess loads
pytest_judge_guard/mh_pb_judge_guard.py, which records every judge call to a per-checkpoint log without editing
the vendored upstream tree; a checkpoint whose verdict passed through a failed judge call is not scored.
Every result carries a `judge` block (role, model, calls, error) so a reader can see who judged it.
"""
import json
import os
import re
import subprocess
import sys
import tempfile

CONN_SIG = re.compile(r"Connection refused|URLError|Max retries|Failed to establish|HTTPError: 5|ConnectionError|database has been closed", re.I)
GUARD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pytest_judge_guard")
GUARD_PLUGIN = "mh_pb_judge_guard"


def _read_judge_log(path):
    rows = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        rows.append({"ok": False, "stage": "log", "error": "unreadable judge log line"})
    except FileNotFoundError:
        pass
    return rows


def _judge_block(rows):
    errors = [r.get("error") for r in rows if not r.get("ok")]
    calls = sum(1 for r in rows if r.get("stage") == "call_llm")
    models = sorted({r.get("model") for r in rows if r.get("model")})
    if not models and os.environ.get("LLM_JUDGE_MODEL"):   # no call was made: the configured judge still judged it
        models = [os.environ["LLM_JUDGE_MODEL"]]
    return {"role": "pb_content", "model": (models[0] if len(models) == 1 else (models or None)),
            "runs": 1, "calls": calls, "error": ("; ".join(str(e) for e in errors)[:500] if errors else None),
            "n_errors": len(errors)}


# A checkpoint's pytest waits on its LLM judge, which the guard retries through rate limits and provider hiccups
# for up to ~15 min (mh_pb_judge_guard: 5-60 s backoff, the responses deadline 900 s). A shorter timeout would cut
# those waits short: the checkpoint would end "pytest timeout" with no judge call made and void its whole task.
PYTEST_TIMEOUT = 900


def run_native_pytest(node_ref, pb_repo, fhir_base, job_dir, timeout=PYTEST_TIMEOUT):
    fd, log = tempfile.mkstemp(prefix="mh_pb_judge_", suffix=".jsonl")
    os.close(fd)
    env = {**os.environ, "FHIR_BASE_URL": fhir_base, "JOB_DIR": job_dir, "MH_PB_JUDGE_LOG": log,
           "PYTHONPATH": os.pathsep.join([GUARD_DIR] + [p for p in [os.environ.get("PYTHONPATH")] if p])}
    try:
        try:
            r = subprocess.run([sys.executable, "-m", "pytest", node_ref, "-q", "--tb=line",
                                "-rA", "-p", "no:cacheprovider", "-p", GUARD_PLUGIN],
                               cwd=pb_repo, env=env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            judge = _judge_block(_read_judge_log(log))
            return {"checkpoint_status": "error", "failure_mode": "environment_error", "note": "pytest timeout",
                    "judge": judge}
        judge = _judge_block(_read_judge_log(log))
    finally:
        try:
            os.unlink(log)
        except OSError:
            pass
    out = (r.stdout or "") + (r.stderr or "")
    rc = r.returncode
    if judge["n_errors"]:
        return {"checkpoint_status": "error", "failure_mode": "judge_error", "judge": judge,
                "note": "judge_error: %s" % judge["error"][:300]}
    if rc == 0:
        return {"checkpoint_status": "passed", "failure_mode": None, "judge": judge}
    if rc >= 2:  # collection / usage / internal error
        return {"checkpoint_status": "error", "failure_mode": "verifier_error", "note": out[-300:], "judge": judge}
    # rc == 1: failures and/or in-test errors
    if CONN_SIG.search(out):
        return {"checkpoint_status": "error", "failure_mode": "environment_error", "note": "FHIR/DB connection",
                "judge": judge}
    if "AssertionError" in out or re.search(r"\bFAILED\b", out):
        first = next((ln for ln in out.splitlines() if "assert" in ln.lower() or "Error" in ln), "")
        return {"checkpoint_status": "failed", "failure_mode": "agent_failure", "note": first[:200], "judge": judge}
    return {"checkpoint_status": "error", "failure_mode": "verifier_error", "note": out[-300:], "judge": judge}
