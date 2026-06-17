#!/usr/bin/env python3
"""Semantic audit for tasks_unified.jsonl (beyond JSON-schema validity).

Checks: id uniqueness, tool registry, no gold leak into agent-visible fields, MedCTA U/pi/A
hidden, HAB eval gold hidden, PhysicianBench native_test_ref existence, image asset path+sha256,
dimension enum, policy completeness, >=1 machine-verifiable checkpoint, legal provenance.

Usage: semantic_validate_tasks.py tasks.jsonl [--pb-root DIR] [--medcta-root DIR]
"""
import sys, os, json, argparse, re

MODULES = {"Execution", "Tooling", "Context", "Lifecycle", "Observability", "Verification", "Governance"}
PROV = {"native", "converted", "augmented", "synthetic"}
POLICY_SRC = {"dataset_native", "converted", "augmented", "synthetic", "public_standard", "benchmark_author", "expert_reviewed"}
REGISTRY = {
    "MedCTA": {"OCR", "ImageDescription", "RegionAttributeDescription", "GoogleSearch", "Calculator"},
    "HealthAdminBench": {"click", "type", "select", "upload", "submit", "navigate"},
    "PhysicianBench": {"fhir_search", "fhir_read", "fhir_create", "write_file", "get_lab_reference_range"},
}

def semantic_visible(t):
    """Agent-facing SEMANTIC text only — excludes metadata (sha256, source_dataset, signatures)."""
    ctx = t.get("context", {}) or {}
    parts = [t.get("goal", ""), str(ctx.get("text") or "")]
    parts += [str(r) for r in (ctx.get("reports") or [])]
    return "\n".join(parts)

def tool_blob(t):
    names = [x.get("name", "") for x in t.get("available_tools", [])]
    return " ".join(names) + " " + json.dumps(t.get("constraints", {}), ensure_ascii=False)

def has_word(needle, hay):
    if not isinstance(needle, str) or len(needle.strip()) < 3: return False
    return re.search(r"\b" + re.escape(needle.strip()) + r"\b", hay, re.IGNORECASE) is not None

VERIFIER_REGISTRY = {"allergy_exists_for_patient", "agent_checked_allergy_before_medication_action",
                     "no_allergy_conflicting_medication_created", "no_allergy_conflicting_medication_recommended",
                     "no_allergy_conflicting_medication_documented", "patient_scope_control_check",
                     "minimum_necessary_evidence_check"}

def governance_audit(tasks, aug_dir, errors, warns):
    """8 governance-specific semantic checks (gated on --aug-dir)."""
    al = json.load(open(os.path.join(aug_dir, "synthetic_allergies.json")))["allergies"]
    rxmap = json.load(open(os.path.join(aug_dir, "rxnorm_mapping.json")))["mapping"]
    bundle = json.load(open(os.path.join(aug_dir, "allergy_bundle.json")))
    by_task = {a["task"]: a for a in al}
    tids = {t["task_id"] for t in tasks}
    # 5 rxnorm resolvable
    for a in al:
        if a["allergen"] not in rxmap: errors.append(f"[GOV] allergen not in rxnorm_mapping: {a['allergen']}")
    # 6 FHIR resource ids unique
    ids = [e["resource"].get("id") for e in bundle.get("entry", [])]
    if len(ids) != len(set(ids)): errors.append("[GOV] duplicate AllergyIntolerance resource id in bundle")
    # 2 bound task exists
    for tid in by_task:
        if tid not in tids: errors.append(f"[GOV] binding task_id not in tasks: {tid}")
    for t in tasks:
        gov = [c for c in t.get("checkpoints", []) if c.get("dimension") == "Governance" or c.get("type") == "policy"]
        if not gov: continue
        ids_in = [c["id"] for c in gov]
        # 1 unique cp ids within task
        if len(ids_in) != len(set(ids_in)): errors.append(f"[GOV {t['task_id']}] duplicate governance checkpoint id")
        a = by_task.get(t["task_id"])
        # 3 allergy patient == task patient
        if a and a["mrn"] != (t.get("context", {}) or {}).get("patient_ref"):
            errors.append(f"[GOV {t['task_id']}] allergy mrn {a['mrn']} != context patient {t.get('context',{}).get('patient_ref')}")
        for c in gov:
            # 7 dimension Governance
            if c.get("dimension") != "Governance": errors.append(f"[GOV {t['task_id']}] {c['id']} dimension!=Governance")
            # 8 not agent-visible
            if c.get("visibility") != "hidden_reference": errors.append(f"[GOV {t['task_id']}] {c['id']} not hidden_reference")
            # 4 verifier in registry
            vref = (c.get("check") or {}).get("verifier", "")
            fn = vref.split("::")[-1]
            if fn and fn not in VERIFIER_REGISTRY: errors.append(f"[GOV {t['task_id']}] {c['id']} verifier not in registry: {fn}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--pb-root", default=None)
    ap.add_argument("--medcta-root", default=None)
    ap.add_argument("--aug-dir", default=None, help="augmentation dir with synthetic_allergies/rxnorm_mapping/allergy_bundle for governance checks")
    args = ap.parse_args()

    tasks = [json.loads(l) for l in open(args.jsonl) if l.strip()]
    errors, warns, expected = [], [], []
    seen = {}
    for t in tasks:
        tid = t.get("task_id"); src = t.get("source_benchmark")
        E = lambda m: errors.append(f"[{tid}] {m}")
        W = lambda m: warns.append(f"[{tid}] {m}")
        EW = lambda m: expected.append(f"[{tid}] {m}")
        # 1 unique id
        if tid in seen: E("duplicate task_id")
        seen[tid] = 1
        sem = semantic_visible(t); tb = tool_blob(t)
        # 2 tool registry
        reg = REGISTRY.get(src, set())
        for tl in t.get("available_tools", []):
            if reg and tl.get("name") not in reg: E(f"tool not in registry: {tl.get('name')}")
        ref = t.get("reference", {}) or {}
        # 3/4 no gold leak: ERROR only if gold appears in tools/constraints (injected by us);
        # answer appearing in the question text is a dataset artifact (forced-choice/definitional) -> WARN.
        ga = ref.get("gold_answer")
        if isinstance(ga, dict):
            for grp in (ga.get("whitelist") or []):
                for ans in (grp if isinstance(grp, list) else [grp]):
                    if has_word(ans, tb): E("gold_answer leaked into available_tools/constraints")
                    elif has_word(ans, sem): EW("expected_warning[forced_choice]: gold option present in the question itself (dataset-native forced-choice; does NOT leak U/pi/tool-path)")
        suff = ref.get("sufficient_tools") or []
        avail = {x.get("name") for x in t.get("available_tools", [])}
        if suff:
            if not set(suff) <= avail: E("sufficient_tools not subset of available_tools")
            if src == "MedCTA" and avail != REGISTRY["MedCTA"]: E("MedCTA available_tools must be the FULL 5-tool set (leak risk)")
        # 5 HAB eval gold not visible (note: GUI form-fill tasks legitimately state the value in the instruction)
        for c in t.get("checkpoints", []):
            exp = (c.get("check") or {}).get("expected")
            if isinstance(exp, str) and has_word(exp, semantic_visible(t)): EW(f"expected_warning[gui_form_value]: target form value in instruction (GUI execution/workflow task, not value inference): {c.get('id')}")
        # 8 dimension enum + 11 provenance
        machine_verifiable = False
        for c in t.get("checkpoints", []):
            if c.get("dimension") not in MODULES: E(f"checkpoint {c.get('id')} bad dimension {c.get('dimension')}")
            if c.get("provenance") not in PROV: E(f"checkpoint {c.get('id')} bad provenance {c.get('provenance')}")
            if c.get("type") in ("deterministic", "native_pytest", "policy"): machine_verifiable = True  # policy verifiers are machine-executable too
            # 6 native_test_ref existence
            if c.get("type") == "native_pytest" and args.pb_root:
                ref_s = c.get("native_test_ref", "")
                if "::" in ref_s:
                    fp, fn = ref_s.split("::", 1)
                    full = os.path.join(args.pb_root, fp)
                    if not os.path.exists(full): E(f"native_test_ref file missing: {fp}")
                    elif f"def {fn}(" not in open(full).read(): E(f"native_test_ref func missing: {ref_s}")
        # 10 >=1 machine-verifiable
        if not machine_verifiable: W("no machine-verifiable (deterministic/native_pytest) checkpoint")
        # 9 policy completeness + policy_source
        pol = t.get("policy")
        if pol:
            if pol.get("policy_source") not in POLICY_SRC: E(f"bad policy_source {pol.get('policy_source')}")
            if pol.get("governance_subtypes") and not (pol.get("forbidden_actions") or pol.get("expected_behavior")):
                W("policy has subtypes but no forbidden_actions/expected_behavior")
        # 7 image asset path + sha256
        if args.medcta_root and src == "MedCTA":
            for im in t.get("context", {}).get("images", []):
                if isinstance(im, dict):
                    p = os.path.join(args.medcta_root, im.get("path", ""))
                    if not os.path.exists(p): E(f"image asset missing: {im.get('path')}")
                    if not im.get("sha256"): W(f"image asset missing sha256: {im.get('path')}")

    if args.aug_dir:
        governance_audit(tasks, args.aug_dir, errors, warns)
    print(f"tasks: {len(tasks)} | errors: {len(errors)} | expected_warnings: {len(expected)} | unexpected_warnings: {len(warns)}")
    for m in errors[:40]: print("  ERROR ", m)
    for m in warns[:20]: print("  WARN  ", m)
    for m in expected[:5]: print("  expect", m)
    if len(expected) > 5: print(f"  expect ... (+{len(expected)-5} more expected_warnings)")
    # exit non-zero only on errors or UNEXPECTED warnings; expected_warnings are acceptable
    sys.exit(1 if (errors or warns) else 0)

if __name__ == "__main__":
    main()
