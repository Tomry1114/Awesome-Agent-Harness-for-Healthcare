#!/bin/bash
cd /hpc2hdd/home/ce483/Medical_harness
PY=~/.conda/envs/medicalharness/bin/python
export MH_OPENAI_BASE=https://www.micuapi.ai MH_API_MODEL=gpt-5.4-mini MH_OPENAI_TIMEOUT=180 PYTHONUNBUFFERED=1
BASE="MH_REPAIR=full MH_REPAIR_CHANNEL=inline MH_HARNESS_MODE=enforce MH_HARNESS_JUDGE_MODEL=gpt-5.4 MH_TOOL_MODE=real MH_PLAN_COMPLETENESS=1"
FB="--fhir-base http://localhost:38080/fhir"; TID="PB-abnormal_uterine_bleeding"
rm -rf rep_rec rep_off rep_on rep.log script.json
echo "[CONTROLLED REPLAY $(date)]" > rep.log

# 1) RECORD the real agent (up to 3 tries) until we get a failure-case trace: writes the deliverable, no order create
GOT=0
for i in 1 2 3; do
  echo "[record attempt $i]" >> rep.log
  rm -rf rep_rec
  env $BASE MH_COMPLETE_EFFECT=0 $PY runner/run_batch.py --bench PhysicianBench --agent gpt5 --task-id $TID $FB --reset-mode restore_pristine --out rep_rec/ >> rep.log 2>&1
  $PY runner/extract_script.py rep_rec/gpt5/$TID/trajectory.jsonl script.json >> rep.log 2>&1 && { GOT=1; break; }
  echo "[record attempt $i: agent placed the order or no write -- retrying]" >> rep.log
done
if [ $GOT -ne 1 ]; then echo "[FAILED to record a no-order trace in 3 tries -- NO attribution sample]" >> rep.log; echo "[e2e ALL DONE $(date)]" >> rep.log; exit 3; fi

# 2) REPLAY OFF (scripted == recorded agent, effect=0) and ON (SAME script, effect=1). Only variable = MH_COMPLETE_EFFECT.
env $BASE MH_COMPLETE_EFFECT=0 MH_SCRIPT_FILE=$PWD/script.json $PY runner/run_batch.py --bench PhysicianBench --agent scripted --task-id $TID $FB --reset-mode restore_pristine --out rep_off/ >> rep.log 2>&1
env $BASE MH_COMPLETE_EFFECT=1 MH_SCRIPT_FILE=$PWD/script.json $PY runner/run_batch.py --bench PhysicianBench --agent scripted --task-id $TID $FB --reset-mode restore_pristine --out rep_on/ >> rep.log 2>&1

# 3) compare -- same agent script, attribute cp3 delta 100% to the harness
$PY - <<'PYX' >> rep.log 2>&1
import json, glob
TID="PB-abnormal_uterine_bleeding"
def load(root):
    f=glob.glob("%s/scripted/%s/result.json"%(root,TID))
    if not f: return None
    d=json.load(open(f)); cps={c["id"]:c["checkpoint_status"] for c in d.get("checkpoints",[])}
    tj=f[0].replace("result.json","trajectory.jsonl"); ev={"effect":[],"agent_create":0,"recovery_create":0}
    import os
    if os.path.exists(tj):
        for l in open(tj):
            try:r=json.loads(l)
            except:continue
            if r.get("event_type")=="effect_completion": ev["effect"].append({k:r.get(k) for k in ("episode_state","reason","created_id")})
            if r.get("event_type")=="tool_call":
                _t=str(r.get("tool","")); _rt=((r.get("args") or {}).get("resource") or {}).get("resourceType")
                if "service_request_create" in _t or (_t=="fhir_create" and _rt=="ServiceRequest"):
                    ev["recovery_create" if r.get("origin")=="recovery" else "agent_create"]+=1
    return {"cps":cps,"ev":ev}
off,on=load("rep_off"),load("rep_on")
print("\n================ CONTROLLED REPLAY (identical agent script; only MH_COMPLETE_EFFECT differs) ================")
for n,r in (("REPLAY-OFF",off),("REPLAY-ON",on)):
    if not r: print(n,"MISSING"); continue
    print("\n---- %s ----"%n)
    print("  cp3:",r["cps"].get("cp3_pelvic_ultrasound_order"),"| agent_creates:",r["ev"]["agent_create"],"| recovery_creates:",r["ev"]["recovery_create"])
    print("  effect:",r["ev"]["effect"])
if off and on:
    ft=[c for c in on["cps"] if off["cps"].get(c)=="failed" and on["cps"][c]=="passed"]
    tf=[c for c in on["cps"] if off["cps"].get(c)=="passed" and on["cps"][c]=="failed"]
    print("\n  ATTRIBUTABLE (same script):  F->T:",ft,"  T->F:",tf)
PYX
echo "[e2e ALL DONE $(date)]" >> rep.log
