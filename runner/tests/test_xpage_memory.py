"""Cross-page memory carries only TRUSTED facts (persisted state / read-only records), resolves a concept to ONE value
(higher provenance wins; conflicting equal-provenance values -> ambiguous -> no auto-bind), and binds facts to the
current SUBJECT so they never cross a patient/case switch. Editable drafts never bind. These pin that a value the
harness fills in from memory comes from a trusted source for the right patient or case."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from workspace import H0, Workspace
W = Workspace(H0["gui"]).module("recovery/gui_completion.py").GenericGuiCompletionWorkflow
from harness.bindings import AUTHORITATIVE_STATE


class _G:
    committed_fields = {}
    raw = {}
    goal_type = "gui"


def _rec(label, value):                                  # a read-only RECORD (trusted displayed fact)
    return {"role": "textbox", "label": label, "value": value, "readonly": True}


def test_read_only_record_recalls_across_synonyms():
    ctx = {}
    W()._remember(ctx, [_rec("Subscriber ID", "AET123456789"), _rec("Date of Birth", "1965-03-15")], {})
    mem = ctx["_scratch_memory"]
    assert mem["member id"]["value"] == "AET123456789"   # recallable via the payer-form synonym
    assert mem["memberid"]["value"] == "AET123456789"
    assert mem["dob"]["value"] == "1965-03-15"


def test_editable_draft_is_not_remembered():
    """An editable input / search box value is a DRAFT (agent-typed, possibly wrong or left over) -> never a fact."""
    ctx = {}
    W()._remember(ctx, [{"role": "searchbox", "label": "Member ID", "value": "WRONG-TYPED"}], {})
    assert "member id" not in ctx.get("_scratch_memory", {})


def test_higher_provenance_replaces_not_first_write_wins():
    """A persisted value (rank 3) replaces an earlier read-only record (rank 2) -- not first-write-wins."""
    ctx = {}
    w = W()
    w._remember(ctx, [_rec("Member ID", "OLD123")], {})
    w._remember(ctx, [], {"memberId": "NEW456"})         # persisted -> higher provenance -> replaces
    assert ctx["_scratch_memory"]["member id"]["value"] == "NEW456"


def test_conflicting_equal_provenance_is_ambiguous_and_does_not_bind():
    """Two DIFFERENT equally-trusted values for a concept -> AMBIGUOUS -> not folded into the source."""
    ctx = {}
    w = W()
    w._remember(ctx, [_rec("Member ID", "AAA")], {})
    w._remember(ctx, [_rec("Subscriber ID", "BBB")], {})  # same rank, different value -> ambiguous
    assert ctx["_scratch_memory"]["member id"]["ambiguous"] is True
    assert "member id" not in w._sources(_G, ctx)[AUTHORITATIVE_STATE]


def test_subject_bound_fact_not_carried_across_subject():
    """A fact seen under patient A must not bind on a surface whose subject is patient B."""
    ctx = {}
    w = W()
    # the subject comes from the current ROUTE, not a whole-state scan
    w._remember(ctx, [_rec("Member ID", "PAT_A")], {}, "URL http://x/emr/referral/REF-A")   # subject A
    w._remember(ctx, [], {}, "URL http://x/emr/referral/REF-A")                             # still A -> binds
    assert "member id" in w._sources(_G, ctx)[AUTHORITATIVE_STATE]
    w._remember(ctx, [], {}, "URL http://x/emr/referral/REF-B")                             # switched to B
    assert "member id" not in w._sources(_G, ctx)[AUTHORITATIVE_STATE]     # A's fact must not carry over
