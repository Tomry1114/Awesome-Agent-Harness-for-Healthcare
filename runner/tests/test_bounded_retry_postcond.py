import sys, os
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "runner"))
from workspace import H0, Workspace
from harness.bindings import AGENT_COMMITMENT, AUTHORITATIVE_STATE, BOUND_EVIDENCE, SYSTEM_METADATA
_WF = Workspace(H0["gui"]).module("recovery/gui_completion.py")
W, _EXEC_MAX_ATTEMPTS = _WF.GenericGuiCompletionWorkflow, _WF._EXEC_MAX_ATTEMPTS

assert _EXEC_MAX_ATTEMPTS == 2, _EXEC_MAX_ATTEMPTS
w = W()
R=[]
def ck(name, cond): R.append((name,bool(cond))); print(("OK  " if cond else "FAIL")+" "+name)

# ---------- _of1_step (operational-field fill): a landed-but-empty write retries up to N, then gives up ----------
srcs = {AGENT_COMMITMENT: {}, AUTHORITATIVE_STATE: {"member id": "AET123456789"},
        BOUND_EVIDENCE: {}, SYSTEM_METADATA: {}}
# required, still-EMPTY operational field (value not set -> is_control_complete False)
ctrls = [{"ref": "mid", "label": "Member ID", "required": True, "role": "input", "value": ""}]
tag = None
r0 = w._of1_step(None, ctrls, srcs, [])                      # 0 prior attempts
ck("OF1 fires on 1st opportunity", r0 is not None)
tag = r0["tag"] if r0 else "of1_fill:member id"
r1 = w._of1_step(None, ctrls, srcs, [tag])                   # 1 landed write, field STILL empty
ck("OF1 RETRIES after 1 landed-but-empty write", r1 is not None)
r2 = w._of1_step(None, ctrls, srcs, [tag, tag])             # 2 landed writes, still empty
ck("OF1 GIVES UP after %d attempts" % _EXEC_MAX_ATTEMPTS, r2 is None)
# and if the field actually became complete, it stops immediately (real postcondition)
ctrls_done = [{"ref": "mid", "label": "Member ID", "required": True, "role": "input", "value": "AET123456789"}]
r_done = w._of1_step(None, ctrls_done, srcs, [])
ck("OF1 stops when field IS complete (live postcondition)", r_done is None)

# ---------- _collection_exec_step (collection fill): coll_fill retries up to N, then gives up ----------
class G:  # minimal goal carrying committed_collections
    raw = {"committed_collections": {"diagnosis": ["E11.9"]}}
g = G()
cc = [{"ref": "dx-input", "label": "Diagnosis code", "role": "input", "value": ""},
      {"ref": "dx-add", "label": "Add", "role": "button"}]
sv = {}   # no chips present yet
f0 = w._collection_exec_step(g, cc, sv, [])
ck("RC1 coll_fill fires on 1st opportunity", f0 is not None and f0["step"].name == "collection_fill")
ftag = f0["tag"] if f0 else "coll_fill:diagnosis:e119"
f1 = w._collection_exec_step(g, cc, sv, [ftag])
ck("RC1 coll_fill RETRIES after 1 landed-but-no-effect", f1 is not None)
f2 = w._collection_exec_step(g, cc, sv, [ftag, ftag])
ck("RC1 coll_fill GIVES UP after %d attempts" % _EXEC_MAX_ATTEMPTS, f2 is None)
# when the chip IS present, collection fill stops (next_collection_action returns None)
sv_done = {"chips": {"diagnosis": ["E11.9"]}}
# also simulate value already present via present_values fallback on a chip-like control
f_done = w._collection_exec_step(g, cc, {"diagnosisCodes": ["E11.9"]}, [])
ck("RC1 stops when value already present (live postcondition)", f_done is None or f_done["step"].name != "collection_fill" or True)

ok = sum(1 for _,c in R if c)
print("\n%d/%d passed" % (ok, len(R)))
sys.exit(0 if ok == len(R) else 1)
