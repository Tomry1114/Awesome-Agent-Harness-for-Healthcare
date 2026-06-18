import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import run
bench, tid = sys.argv[1], sys.argv[2]
mx = int(sys.argv[3]) if len(sys.argv) > 3 else 10
res = run.run_task(bench, tid, "qwen", max_steps=mx, cleanup=False)
traj = []
for ev in res.get("_trajectory", []):
    traj.append({"step": ev.get("step"), "type": ev.get("event_type"), "tool": ev.get("tool"),
                 "args": ev.get("args"), "obs": (str(ev.get("result"))[:400] if ev.get("result") else None),
                 "final": ev.get("thought")})
out = {"task": res["task_id"], "success": res["success"], "evaluation_status": res["evaluation_status"],
       "dimension_scores": res["dimension_scores"], "proxy_dimension_scores": res.get("proxy_dimension_scores"),
       "checkpoints": [(c["id"], c["checkpoint_status"]) for c in res["checkpoints"]],
       "schema": res.get("_schema"), "trajectory": traj}
dst = os.path.join(os.path.dirname(__file__), "agent_%s_%s.json" % (bench, tid))
json.dump(out, open(dst, "w"), ensure_ascii=False, indent=1)
print("WROTE", dst, "| success", res["success"], "| status", res["evaluation_status"])
