cd /hpc2hdd/home/ce483/Medical_harness
PY=~/.conda/envs/medicalharness/bin/python
export PYTHONUNBUFFERED=1
BASE="MH_REPAIR=full MH_HARNESS_MODE=enforce MH_TOOL_MODE=real"
TID="HAB-denial-easy-1"
rm -rf hab_off hab_on hab.log
echo "[HAB controlled replay (mock GUI, scripted; only MH_COMPLETE_EFFECT differs) $(date)]" > hab.log
env $BASE MH_COMPLETE_EFFECT=0 MH_SCRIPT_FILE=$PWD/hab_script.json $PY runner/run_batch.py --bench HealthAdminBench --agent scripted --task-id $TID --out hab_off/ --max-steps 30 >> hab.log 2>&1
env $BASE MH_COMPLETE_EFFECT=1 MH_SCRIPT_FILE=$PWD/hab_script.json $PY runner/run_batch.py --bench HealthAdminBench --agent scripted --task-id $TID --out hab_on/ --max-steps 30 >> hab.log 2>&1
$PY - <<'PYX' >> hab.log 2>&1
import json, glob, os
TID="HAB-denial-easy-1"
def load(root):
    f=glob.glob("%s/scripted/%s/result.json"%(root,TID))
    if not f: return None
    d=json.load(open(f[0])); cps={c["id"]:c["checkpoint_status"] for c in d.get("checkpoints",[])}
    tj=f[0].replace("result.json","trajectory.jsonl"); ev={"effect":[],"agent_doc":0,"recovery_doc":0}
    if os.path.exists(tj):
        for l in open(tj):
            try:r=json.loads(l)
            except:continue
            if r.get("event_type")=="effect_completion": ev["effect"].append({k:r.get(k) for k in ("episode_state","reason","auth_status")})
            if r.get("event_type")=="tool_call" and (r.get("args") or {}).get("target")=="documentedAppealInEpic":
                ev["recovery_doc" if r.get("origin")=="recovery" else "agent_doc"]+=1
    return {"cps":cps,"ev":ev}
off,on=load("hab_off"),load("hab_on")
print("\n======== HAB CONTROLLED REPLAY (mock GUI; identical scripted agent; only MH_COMPLETE_EFFECT differs) ========")
for n,r in (("OFF(effect=0)",off),("ON(effect=1)",on)):
    if not r: print(n,"MISSING"); continue
    print("\n---- %s ----"%n)
    print("  cp2_disposition:",r["cps"].get("cp2_task_resolution"),"| cp3_documented:",r["cps"].get("cp3_documentation"))
    print("  agent_doc:",r["ev"]["agent_doc"],"| recovery_doc:",r["ev"]["recovery_doc"],"| effect:",r["ev"]["effect"])
if off and on:
    ft=[c for c in on["cps"] if off["cps"].get(c)=="failed" and on["cps"][c]=="passed"]
    tf=[c for c in on["cps"] if off["cps"].get(c)=="passed" and on["cps"][c]=="failed"]
    print("\n  ATTRIBUTABLE (same script):  F->T:",ft,"  T->F:",tf)
PYX
echo "[HAB DONE $(date)]" >> hab.log
