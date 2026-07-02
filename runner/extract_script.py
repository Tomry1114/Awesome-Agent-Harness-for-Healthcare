"""Extract the AGENT's OWN recorded action sequence (oracle-blind -- NOT the gold trace) from a trajectory into
a ScriptedAgent script, so OFF and ON can replay the IDENTICAL agent behavior. exit 0 = a usable failure-case
trace (has the deliverable write, but NO order create); exit 3 = agent already placed the order (not the target)."""
import json, sys

def _is_service_request_create(tool, args):
    t = str(tool or "")
    if "service_request_create" in t:
        return True
    if t == "fhir_create":
        return ((args or {}).get("resource") or {}).get("resourceType") == "ServiceRequest"
    return False

tj, out = sys.argv[1], sys.argv[2]
steps, has_sr_create, has_write = [], False, False
for l in open(tj):
    try:
        r = json.loads(l)
    except Exception:
        continue
    if r.get("event_type") == "tool_call" and r.get("origin") != "recovery":   # agent actions only
        tool, args = r.get("tool"), (r.get("args") or {})
        steps.append({"type": "tool_call", "tool": tool, "args": args})
        if _is_service_request_create(tool, args):   # generic fhir_create(resourceType=ServiceRequest) OR granular
            has_sr_create = True
        if tool == "write_file":
            has_write = True
steps.append({"type": "final", "answer": "replay complete"})
json.dump(steps, open(out, "w"))
print("extracted steps=%d has_service_request_create=%s has_deliverable_write=%s"
      % (len(steps), has_sr_create, has_write))
sys.exit(0 if (has_write and not has_sr_create) else 3)
