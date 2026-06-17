"""native_pytest executor: run a PhysicianBench checkpoint (tests/test_outputs.py::func) the same way
upstream scripts/run_eval.py does — subprocess pytest with FHIR_BASE_URL + JOB_DIR env, cwd=repo.

Maps the outcome to (checkpoint_status, failure_mode):
  rc 0            -> passed
  rc 1 + Assertion-> failed / agent_failure
  rc 1 + conn err -> error  / environment_error
  rc 1 + other exc-> error  / verifier_error
  rc >=2          -> error  / verifier_error   (collection/usage/internal)
"""
import os, sys, subprocess, re

CONN_SIG = re.compile(r"Connection refused|URLError|Max retries|Failed to establish|HTTPError: 5|ConnectionError|database has been closed", re.I)

def run_native_pytest(node_ref, pb_repo, fhir_base, job_dir, timeout=150):
    env = {**os.environ, "FHIR_BASE_URL": fhir_base, "JOB_DIR": job_dir}
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", node_ref, "-q", "--tb=line",
                            "-rA", "-p", "no:cacheprovider"],
                           cwd=pb_repo, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"checkpoint_status": "error", "failure_mode": "environment_error", "note": "pytest timeout"}
    out = (r.stdout or "") + (r.stderr or "")
    rc = r.returncode
    if rc == 0:
        return {"checkpoint_status": "passed", "failure_mode": None}
    if rc >= 2:  # collection / usage / internal error
        return {"checkpoint_status": "error", "failure_mode": "verifier_error", "note": out[-300:]}
    # rc == 1: failures and/or in-test errors
    if CONN_SIG.search(out):
        return {"checkpoint_status": "error", "failure_mode": "environment_error", "note": "FHIR/DB connection"}
    if "AssertionError" in out or re.search(r"\bFAILED\b", out):
        first = next((ln for ln in out.splitlines() if "assert" in ln.lower() or "Error" in ln), "")
        return {"checkpoint_status": "failed", "failure_mode": "agent_failure", "note": first[:200]}
    return {"checkpoint_status": "error", "failure_mode": "verifier_error", "note": out[-300:]}
