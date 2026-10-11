"""HAB cross-page memory test: a value SEEN on an earlier surface (EHR) must be BINDABLE on a later
surface (payer-portal form) instead of BLOCKing as needs-new-content -- HAB's hidden long-term dependency.
Standalone: sys.exit(0) on pass. No model, no GPU."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workspace import H0, Workspace

GenericGuiCompletionWorkflow = Workspace(H0["gui"]).module("recovery/gui_completion.py").GenericGuiCompletionWorkflow

fails = []


class _Goal:
    committed_fields = {}
    raw = {"verify_spec": {"path": "done", "check": "truthy"}}
    goal_type = "gui"


w = GenericGuiCompletionWorkflow()
ctx = {"authoritative_state": {}, "system_metadata": {}}

# PAGE 1 (EHR surface): a Member ID is visible as a filled control -> enters cross-page memory.
p1 = [{"role": "input", "name": "Member ID", "label": "Member ID", "value": "MBR-99123", "readonly": True}]  # READ-ONLY record
w.next_action(_Goal, ctx, "", p1, {}, [])
if "member id" not in ctx.get("_scratch_memory", {}):
    fails.append("P0-4: a value seen on page 1 did not enter the scratch memory")

# PAGE 2 (payer portal): an EMPTY required Member ID field -> memory must BIND it (a FILL step), not BLOCK.
p2 = [{"role": "input", "name": "Member ID", "label": "Member ID", "value": "", "required": True},
      {"role": "button", "label": "Submit", "commit": True}]
d = w.next_action(_Goal, ctx, "", p2, {}, [])
if d.get("kind") != "step":
    fails.append("P0-4: page-2 required field BLOCKED instead of binding from memory (%s)" % d.get("reason"))
elif d["step"].action.get("value") != "MBR-99123":
    fails.append("P0-4: page-2 field bound the wrong value: %r" % d["step"].action.get("value"))

# NEGATIVE control: a field whose value was NEVER seen must still BLOCK (needs new content, not a hallucination).
ctx2 = {"authoritative_state": {}, "system_metadata": {}}
p3 = [{"role": "textarea", "name": "Appeal Rationale", "label": "Appeal Rationale", "value": "", "required": True},
      {"role": "button", "label": "Submit", "commit": True}]
d3 = w.next_action(_Goal, ctx2, "", p3, {}, [])
if d3.get("kind") != "block":
    fails.append("P0-4: a never-seen required field must BLOCK (needs new content), got %s" % d3.get("kind"))

if fails:
    print("FAIL (%d):" % len(fails))
    for x in fails:
        print("  -", x)
    sys.exit(1)
print("PASS: cross-page memory binds seen values; never-seen fields still BLOCK (existing-info vs new-content)")
sys.exit(0)
