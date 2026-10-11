"""unresolved_operational_commit (the process_output_inconsistency finalize-blocker) must gate ONLY
IRREVERSIBLE clinical mutations (orders). A REVERSIBLE write (a patient Communication/message) whose
postcondition is unverifiable (verified=None) must NOT trap the agent in a before_final REVISE loop -- it was
landing-confirmed by reconciliation and is amendable. Orders (irreversible) still gate. Missing effect ->
treated as irreversible (safe default)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness.state import Ledger

ok = True
def check(n, c):
    global ok; ok = ok and bool(c); print(("  PASS " if c else "  FAIL ") + n)

# 1) an IRREVERSIBLE order create that did NOT verify (None) -> STILL a finalize-blocking unresolved commit
L = Ledger()
L.record_commit("verify_commit", 1, verified=None, semantic_type="create", effect="irreversible", detail="order")
uoc = L.unresolved_operational_commit()
check("irreversible order, verified=None -> unresolved (still gated)", uoc is not None and uoc.get("effect") == "irreversible")

# 2) a REVERSIBLE communication create that did NOT verify (None) -> NOT a finalize-blocking commit
L2 = Ledger()
L2.record_commit("verify_commit", 1, verified=None, semantic_type="create", effect="reversible", detail="message")
check("reversible communication, verified=None -> NOT unresolved (no finalize block)", L2.unresolved_operational_commit() is None)

# 3) a REVERSIBLE write that verified True is obviously fine
L3 = Ledger()
L3.record_commit("verify_commit", 1, verified=True, semantic_type="create", effect="reversible")
check("reversible verified True -> not unresolved", L3.unresolved_operational_commit() is None)

# 4) missing effect (a caller that passes none) -> SAFE DEFAULT: treated as gating (irreversible)
L4 = Ledger()
L4.record_commit("verify_commit", 1, verified=None, semantic_type="create")   # no effect kwarg
check("missing effect, verified=None -> still gated (safe default)", L4.unresolved_operational_commit() is not None)

# 5) mixed: irreversible order (None) + reversible message (None) -> the ORDER is what gates
L5 = Ledger()
L5.record_commit("verify_commit_msg", 1, verified=None, semantic_type="create", effect="reversible")
L5.record_commit("verify_commit_order", 2, verified=None, semantic_type="create", effect="irreversible")
uoc5 = L5.unresolved_operational_commit()
check("mixed -> only the irreversible order gates (message ignored)", uoc5 is not None and uoc5.get("effect") == "irreversible")

# 6) reversible message that FAILED transport is handled by commit_execution_failed elsewhere, not here:
#    an irreversible order that verified True -> resolved, not gating
L6 = Ledger()
L6.record_commit("verify_commit", 1, verified=True, semantic_type="create", effect="irreversible")
check("irreversible order verified True -> resolved (not gated)", L6.unresolved_operational_commit() is None)

print("\nRESULT:", "ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
