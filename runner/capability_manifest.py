#!/usr/bin/env python3
"""Capability manifest + precheck (Canonical Contract §5). Each environment declares what it can do;
each task's required capabilities are inferred; before running we check required ⊆ available. A task
needing a capability the env lacks is marked `environment_capability_missing` (NOT an agent failure)
and excluded from capability scoring — it enters the integrity report. Prevents fail-by-construction
(e.g. HAB tasks needing file upload/download on an env without them)."""
import json, os, re, sys, collections

ENV_CAPABILITIES = {
    "physicianbench": {
        "tools": ["fhir_search", "fhir_read", "fhir_create", "get_lab_reference_range"],
        "file_operations": ["write"],
        "observation_modalities": ["structured"],
    },
    "healthadminbench": {
        "gui_operations": ["navigate", "click", "type", "select", "scroll", "back", "submit", "snapshot"],
        "file_operations": [],  # upload/download NOT yet implemented (stage-2) -> declared missing
        "observation_modalities": ["text", "structured_elements"],
    },
    "medcta": {
        "tools": ["OCR", "ImageDescription", "RegionAttributeDescription", "GoogleSearch", "Calculator"],
        "file_operations": [],
        "observation_modalities": ["image_ref", "text"],
    },
}
ENV_ALIAS = {"fhir": "physicianbench", "gui": "healthadminbench", "tool_sandbox": "medcta",
             "PhysicianBench": "physicianbench", "HealthAdminBench": "healthadminbench", "MedCTA": "medcta"}


def infer_required(task):
    """Return set of capability tokens this task needs. File ops inferred from explicit verbs in the
    goal/checkpoints (NOT 'submit', which is a generic GUI action present in almost every HAB task)."""
    req = set()
    text = (task.get("goal") or task.get("instruction") or "")
    for cp in (task.get("checkpoints") or []):
        text += " " + json.dumps(cp.get("check") or {}, ensure_ascii=False)
    low = text.lower()
    if re.search(r"\bdownload(ed|ing|s)?\b|retriev\w+ the (auth|denial|remittance|document|letter|pdf)", low):
        req.add("file.download")
    if re.search(r"\bupload(ed|ing|s)?\b|\battach(ed|ing|ment|es)?\b", low):
        req.add("file.upload")
    return req


def precheck(task, env_type):
    env = ENV_ALIAS.get(env_type, env_type)
    caps = ENV_CAPABILITIES.get(env, {})
    have = set("file." + f for f in caps.get("file_operations", [])) \
        | set("gui." + g for g in caps.get("gui_operations", [])) \
        | set(caps.get("tools", []))
    missing = sorted(infer_required(task) - have)
    return {"ok": not missing, "missing_capabilities": missing,
            "qualification": "environment_capability_missing" if missing else None}


if __name__ == "__main__":
    # accurate HAB capability audit (replaces the 'submit'-polluted regex)
    bench = sys.argv[1] if len(sys.argv) > 1 else "HealthAdminBench"
    path = "benchmark_dataprocess/%s/tasks_unified.jsonl" % bench
    cnt = collections.Counter()
    miss = collections.Counter()
    n = 0
    for l in open(path):
        t = json.loads(l)
        n += 1
        req = infer_required(t)
        for r in req:
            cnt[r] += 1
        pc = precheck(t, bench)
        if not pc["ok"]:
            for m in pc["missing_capabilities"]:
                miss[m] += 1
    excluded = sum(1 for l in open(path) if not precheck(json.loads(l), bench)["ok"])
    print("== %s capability audit (n=%d) ==" % (bench, n))
    print("required by tasks:", dict(cnt))
    print("MISSING on current env (-> not_exercised_due_to_missing_capability, NOT agent fail):", dict(miss))
    print("fully runnable: %d/%d  |  capability-excluded: %d/%d" % (n - excluded, n, excluded, n))
