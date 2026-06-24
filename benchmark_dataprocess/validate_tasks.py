#!/usr/bin/env python3
"""Validate a tasks_unified.jsonl against spec/task.schema.json (resolving cross-file $ref)."""
import sys, json, glob, os
from jsonschema import Draft7Validator, RefResolver

SPEC_DIR = os.path.join(os.path.dirname(__file__), "..", "spec")

def load_store():
    store = {}
    for p in glob.glob(os.path.join(SPEC_DIR, "*.json")):
        s = json.load(open(p))
        if "$id" in s:
            store[s["$id"]] = s
    return store

def main():
    jsonl = sys.argv[1]
    store = load_store()
    task_schema = json.load(open(os.path.join(SPEC_DIR, "task.schema.json")))
    resolver = RefResolver(base_uri=task_schema["$id"], referrer=task_schema, store=store)
    validator = Draft7Validator(task_schema, resolver=resolver)
    n = ok = 0
    tasks = []
    for line in open(jsonl):
        line = line.strip()
        if not line: continue
        n += 1
        task = json.loads(line); tasks.append(task)
        errs = sorted(validator.iter_errors(task), key=lambda e: e.path)
        if errs:
            print(f"[FAIL] {task.get('task_id')}")
            for e in errs[:5]:
                print(f"   - {list(e.path)}: {e.message}")
        else:
            ok += 1
    print(f"\nvalidated {ok}/{n} tasks OK against spec/task.schema.json")
    # registry membership (supported values live in spec/registry.json, NOT the schema enum) -- a 4th
    # dataset must register its prefix/domain/modality/environment here to pass strict validation.
    try:
        reg = json.load(open(os.path.join(SPEC_DIR, "registry.json")))
        benches = reg.get("benchmarks", {}); adapters = set(reg.get("environment_adapters", []))
        unreg = []
        for t in tasks:
            sb = t.get("source_benchmark")
            if sb not in benches:
                unreg.append((t.get("task_id"), "unregistered source_benchmark=%s" % sb)); continue
            pref = benches[sb].get("prefix")
            if pref and not str(t.get("task_id", "")).startswith(pref + "-"):
                unreg.append((t.get("task_id"), "task_id prefix != registry prefix %s" % pref))
            et = (t.get("environment") or {}).get("type")
            if et and et not in adapters:
                unreg.append((t.get("task_id"), "unregistered environment adapter=%s" % et))
        if unreg:
            print("REGISTRY: %d task(s) not registered in spec/registry.json:" % len(unreg))
            for tid, why in unreg[:10]: print("  - %s: %s" % (tid, why))
        else:
            print("registry: all %d tasks registered (benchmarks=%s)" % (len(tasks), sorted(benches)))
    except FileNotFoundError:
        print("registry: spec/registry.json missing (skipped)")

    sys.exit(0 if ok == n else 1)

if __name__ == "__main__":
    main()
