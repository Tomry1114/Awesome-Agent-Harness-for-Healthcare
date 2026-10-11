"""MedCTA default-workspace middleware: the answer-candidate strategy.

When the kernel's before_final asks for a targeted observation of the answer (ACQUIRE) or raises a candidate review
(a REVISE whose extra carries `candidate`), the agent's earliest answer is held as root A together with the evidence
ids that existed before the intervention, and the agent's next final becomes candidate B. B is re-verified once. A
residual gap earns one more bounded round, another read or a critique-driven re-answer; otherwise the adoption gates
below judge A against B, and the shell delivers the chosen text: B (or a claim-level merge of A and B that verifies)
or A (flagged when A does not verify).

The module holds the candidate consumption, the hold and agent text of the before_final ACQUIRE and candidate REVISE
paths, and the adoption gates. The hooks at the bottom are how the shell (runner/middleware_host.py) calls it;
what their results do to the loop is the shell's. The round bounds are the workspace's allocation (manifest
semantic_budget.rounds acquire / revise, read as ctx.semantic_rounds). Only MedCTA holds this file: no PhysicianBench or
HealthAdminBench before_final producer returns an ACQUIRE or a candidate REVISE.

The gates judge through ctx.semantic_check(): one check from this middleware's semantic pools for the adoption
gates of one candidate, another for the claim merge, each taken after the verify_final before it. A refused check
leaves the gates without a judge, which is their no-judge path: evaluate_candidate keeps A, and the merge
builds nothing so the reverified B is delivered.
"""
import json

from workspace_api import next_action_of
from workspace_api import as_bool as _as_bool, format_evidence as _format_evidence, json_obj as _json_obj, verify_claim_support
from workspace_api import Accept, Read, Repair


# ---------------------------------------------------------------------------------------------------------
# CONSERVATIVE A/B SELECTOR. Compares the ORIGINAL answer A with a REVISED candidate B on PUBLIC dimensions
# only (no gold, no hidden reference). The adopt decision is made by the caller and is conservative by
# construction: B replaces A ONLY when the judge prefers B with a margin >= tau, the original critique is
# resolved, and B introduces NO new hard violation. Close / uncertain -> keep A (controls T->F harm).
# ---------------------------------------------------------------------------------------------------------
_COMPARE_PROMPT = (
    "You are comparing TWO candidate FINAL ANSWERS (ORIGINAL and REVISED) to the SAME public task, using the "
    "agent's validated evidence. Do NOT solve the task yourself or infer a hidden reference answer. Judge ONLY "
    "on these public dimensions: (1) directly answers the public question; (2) consistency with the validated "
    "evidence; (3) absence of evidence contradiction; (4) process-output consistency; (5) absence of "
    "unsupported specificity; (6) whether the stated CRITIQUE of the original is resolved. Decide which "
    "candidate is better OVERALL, by how much (0=identical, 1=decisive), and whether the REVISED one "
    "introduces any NEW hard violation (contradiction / false success / wrong subject). Be CONSERVATIVE: if "
    "they are close or you are unsure, answer 'uncertain'. Reply STRICT JSON: {{\"preferred\": "
    '"original|revised|uncertain", "margin": 0.0-1.0, "critique_resolved": true|false, '
    '"revised_new_hard_violation": true|false, "reason": "<short>"}}.\n\n'
    "PUBLIC QUESTION/GOAL:\n{task_goal}\n\nPUBLIC CONTEXT:\n{public_context}\n\nVALIDATED EVIDENCE:\n"
    "{evidence}\n\nCRITIQUE OF ORIGINAL:\n{critique}\n\nORIGINAL ANSWER (A):\n{original}\n\n"
    "REVISED ANSWER (B):\n{revised}\n"
)


def compare_answer_candidates(task_goal, public_context, original, revised, critique, evidence, judge_fn=None):
    """Public-dimension A/B comparison -> dict{preferred, margin, critique_resolved, revised_new_hard_violation,
    reason}. No judge / unparseable -> conservative default (preferred='uncertain') so the caller keeps A."""
    default = {"preferred": "uncertain", "margin": 0.0, "critique_resolved": False,
               "revised_new_hard_violation": False, "reason": "comparator_unavailable"}
    if not judge_fn:
        return default
    prompt = _COMPARE_PROMPT.format(
        task_goal=str(task_goal or "(not provided)")[:2000], public_context=str(public_context or "(none)")[:2000],
        evidence=_format_evidence(evidence)[:6000], critique=str(critique or "(none)")[:1000],
        original=str(original)[:2000], revised=str(revised)[:2000])
    try:
        raw = judge_fn(prompt)
    except Exception as ex:
        return dict(default, reason="comparator_error:%r" % (ex,))
    s = raw if isinstance(raw, str) else str(raw)
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        try:
            d = json.loads(s[i:j + 1])
            pref = str(d.get("preferred", "uncertain")).strip().lower()
            if pref not in ("original", "revised", "uncertain"):
                pref = "uncertain"
            return {"preferred": pref, "margin": float(d.get("margin", 0.0) or 0.0),
                    "critique_resolved": _as_bool(d.get("critique_resolved")),
                    "revised_new_hard_violation": _as_bool(d.get("revised_new_hard_violation")),
                    "reason": d.get("reason")}
        except Exception:
            pass
    return default


def adopt_revised(comparison, tau=0.15):
    """CONSERVATIVE adopt rule: replace A with B ONLY if the judge prefers B by >= tau, the critique is
    resolved, and B adds no new hard violation. Anything else -> keep A. (tau is the safety margin.)"""
    c = comparison or {}
    return bool(c.get("preferred") == "revised" and (c.get("margin") or 0.0) >= tau
                and c.get("critique_resolved") and not c.get("revised_new_hard_violation"))


# =====================================================================================================
# STRENGTHENED ANSWER RETENTION (epistemic non-degradation). The public-dimension comparator alone CANNOT
# protect a core clinical conclusion: a more "complete / fluent / critique-resolving" B is NOT evidence
# that the conclusion is more CORRECT (a wrong B can pass both the comparator and before_final). Hard
# rule: a B may FLIP the core decision ONLY when NEW validated evidence (ACQUIRE/CONSULT) directly supports
# that specific flip; a no-new-evidence REVISE may only refine (rationale / format / hedging / removing
# unsupported add-ons) WITHOUT changing the core decision. Oracle-blind throughout; every uncertain branch
# is fail-SAFE (keep A).
# =====================================================================================================

_CORE_SLOTS = ("requested_operation", "primary_conclusion", "polarity", "target",
               "severity_or_priority", "recommended_action")

_SIGNATURE_PROMPT = (
    "Extract the CORE decision expressed by the ANSWER to the PUBLIC task. Do NOT judge correctness or infer "
    "any hidden reference; fill ONLY what the answer actually states, null when absent. Reply STRICT JSON: "
    '{{"requested_operation": "<verb/op or null>", "primary_conclusion": "<main finding/diagnosis/answer>", '
    '"polarity": "present|absent|positive|negative|yes|no|null", "target": "<subject/site/entity or null>", '
    '"severity_or_priority": "<routine|urgent|... or null>", "recommended_action": "<action/disposition or '
    'null>"}}.\n\nPUBLIC GOAL:\n{goal}\n\nANSWER:\n{answer}\n')

_SIGCMP_PROMPT = (
    "Two DECISION SIGNATURES (A, B) were extracted from two candidate answers to the SAME task. Decide whether "
    "the CORE decision CHANGED A->B. CORE change = different primary_conclusion, flipped polarity "
    "(present<->absent), different target/site, different recommended_action/disposition, or a severity change "
    "that alters management (routine->urgent). NOT core: added rationale, added hedging/uncertainty, reworded "
    "phrasing, an added differential that keeps the SAME primary, dose-unit clarification with the same intent. "
    "Be STRICT; if genuinely UNSURE set uncertain=true. Reply STRICT JSON: "
    '{{"changed": true|false, "changed_slots": ["<slot>"...], "uncertain": true|false, "reason": "<short>"}}.'
    "\n\nSIGNATURE A:\n{a}\n\nSIGNATURE B:\n{b}\n")

_DELTA_SUPPORT_PROMPT = (
    # Judge CONTRIBUTION to the A-vs-B decision, NOT whether the new evidence ALONE proves B. A
    # discriminating observation rarely single-handedly proves a full diagnosis; requiring that would reject
    # genuinely useful acquisitions. This is stage 1 (delta contribution); stage 2 (total support over all
    # validated evidence) separately confirms B is actually supported.
    "An answer's CORE decision changed from ORIGINAL (A) to REVISED (B). You are given ONLY the NEWLY ACQUIRED "
    "evidence (not prior evidence, not outside knowledge). Do NOT judge whether this new evidence ALONE proves "
    "B. Judge its CONTRIBUTION to deciding between A and B. Answer supported=true IFF the new evidence BOTH: "
    "(1) addresses the specific discriminating feature / region / attribute at issue between A and B, AND "
    "(2) materially shifts the relative support from A toward B (makes B more supported and/or A less supported "
    "than before) -- even if OTHER evidence is still needed to fully establish B. Answer supported=false if the "
    "new evidence is OFF-TARGET (does not address the discriminator) or does NOT change the A-vs-B balance. "
    "Cite the evidence ids (e.g. ev-7) that contribute. Reply STRICT JSON: "
    '{{"supported": true|false, "changed_slots": [...], "supporting_evidence_ids": ["<id>"...], '
    '"confidence": 0.0-1.0, "reason": "<short>"}}.\n\nPUBLIC GOAL:\n{goal}\n\nORIGINAL CORE SIGNATURE:\n{a}\n\n'
    "REVISED CORE SIGNATURE:\n{b}\n\nNEWLY ACQUIRED EVIDENCE (only):\n{ev}\n")

_PRESERVE_PROMPT = (
    "ORIGINAL answer A and REVISED answer B for the same task. The revision was asked to address ONLY this "
    "critique: {critique}. Decide whether B REMOVED or CONTRADICTED substantive content present in A that was "
    "NOT the subject of that critique (collateral damage). Reply STRICT JSON: "
    '{{"preserved": true|false, "removed": "<what, short>"}}.\n\nA:\n{a}\n\nB:\n{b}\n')


def _format_new_evidence(evidence):
    out = []
    for e in (evidence or []):
        if isinstance(e, dict):
            out.append("[%s] [%s] %s" % (e.get("evidence_id", "ev-?"), e.get("type", "evidence"),
                                         str(e.get("value_full") or e.get("value", ""))[:1500]))
        else:
            out.append(str(e)[:300])
    return "\n".join(out)


def extract_decision_signature(goal_spec, answer, judge_fn=None):
    """Oracle-blind: structured CORE decision of an answer (or None on no-judge/parse-fail)."""
    if not judge_fn:
        return None
    goal = goal_spec.get("goal") if isinstance(goal_spec, dict) else goal_spec
    prompt = _SIGNATURE_PROMPT.format(goal=str(goal or "(none)")[:1500], answer=str(answer or "")[:2000])
    try:
        d = _json_obj(judge_fn(prompt))
    except Exception:
        return None
    if not isinstance(d, dict):
        return None
    return {k: (str(d.get(k)).strip() if d.get(k) is not None else None) for k in _CORE_SLOTS}


def compare_decision_signatures(sig_a, sig_b, judge_fn=None):
    """Did the CORE decision change A->B? SEPARATE judge from the public comparator (a B-preferring comparator
    must not get to also rule the flip 'immaterial'). Fail-SAFE: missing sig / no judge / parse-fail / uncertain
    -> changed=True."""
    default = {"changed": True, "changed_slots": ["__unknown__"], "uncertain": True, "reason": "sigcmp_unavailable"}
    if sig_a is None or sig_b is None or not judge_fn:
        return default
    if sig_a == sig_b:
        return {"changed": False, "changed_slots": [], "uncertain": False, "reason": "identical_signature"}
    prompt = _SIGCMP_PROMPT.format(a=json.dumps(sig_a, ensure_ascii=False), b=json.dumps(sig_b, ensure_ascii=False))
    try:
        d = _json_obj(judge_fn(prompt))
    except Exception:
        return default
    if not isinstance(d, dict):
        return default
    unc = _as_bool(d.get("uncertain"))
    return {"changed": _as_bool(d.get("changed")) or unc, "changed_slots": list(d.get("changed_slots") or []),
            "uncertain": unc, "reason": d.get("reason")}


def verify_change_against_delta(goal, sig_a, sig_b, new_evidence, judge_fn=None):
    """Stage 1: does the newly-acquired evidence CONTRIBUTE to the A->B decision -- i.e. address the
    discriminator AND shift relative support toward B -- NOT whether it alone proves B. Stage 2 (total
    support) confirms B separately. Fail-SAFE: no judge / no new evidence / parse-fail -> supported=False."""
    default = {"supported": False, "changed_slots": [], "supporting_evidence_ids": [], "confidence": 0.0,
               "reason": "delta_support_unavailable"}
    if not judge_fn or not new_evidence:
        return default
    prompt = _DELTA_SUPPORT_PROMPT.format(goal=str(goal or "")[:1200], a=json.dumps(sig_a, ensure_ascii=False),
                                          b=json.dumps(sig_b, ensure_ascii=False),
                                          ev=_format_new_evidence(new_evidence)[:5000])
    try:
        d = _json_obj(judge_fn(prompt))
    except Exception:
        return default
    if not isinstance(d, dict):
        return default
    return {"supported": _as_bool(d.get("supported")),
            "supporting_evidence_ids": [str(x) for x in (d.get("supporting_evidence_ids") or [])],
            "confidence": float(d.get("confidence", 0.0) or 0.0),
            "changed_slots": list(d.get("changed_slots") or []), "reason": d.get("reason")}


_HEDGE_PROMPT = (
    "ORIGINAL answer A and REVISED answer B for the SAME clinical/perceptual question. Decide whether B is a "
    "HEDGE / DEGRADATION of A: B is LESS committal or LESS informative -- it drops a specific finding A made, "
    "retreats to 'no clear difference' / 'cannot determine' / 'no X seen', or becomes vaguer -- WITHOUT adding "
    "any NEW specific finding or correcting a stated error. If B instead adds/sharpens a finding, corrects A, "
    "or is at least as specific, answer hedge=false. Reply STRICT JSON: {{\"hedge\": true|false, \"why\": \"<short>\"}}."
    "\n\nA:\n{a}\n\nB:\n{b}\n")

_REINTERP_PROMPT = (
    "ORIGINAL answer A asserted a finding/conclusion about a medical image; REVISED answer B reaches a "
    "CONTRADICTORY conclusion about the SAME image finding. You are given the NEWLY ACQUIRED evidence. Decide "
    "whether B's contradiction rests on INDEPENDENT NEW evidence -- a genuinely DIFFERENT observation (a new "
    "region, a new attribute, external reference knowledge) that RESOLVES the question -- OR whether it is "
    "merely a DIFFERENT INTERPRETATION of the SAME feature A was about: a PERCEPTUAL AMBIGUITY where one image "
    "finding is read two OPPOSITE ways (present vs absent, or one structure/tissue identity vs a different one), "
    "with NO independent new evidence that actually settles it. If it is a "
    "re-interpretation of the SAME finding WITHOUT independent resolving evidence, reply ambiguous=true (the "
    "harness cannot know which reading is correct, so it must not flip). Reply STRICT JSON: "
    "{{\"ambiguous\": true|false, \"why\": \"<short>\"}}.\n\nA:\n{a}\n\nB:\n{b}\n\nNEWLY ACQUIRED EVIDENCE:\n{ev}\n")

_FLIPDROP_PROMPT = (
    "ORIGINAL answer A and REVISED answer B are both about the same medical image; B changes the core "
    "conclusion. Using the VALIDATED EVIDENCE, decide whether adopting B would LOSE information: does B DROP or "
    "OMIT a specific EVIDENCE-SUPPORTED finding/slot that A correctly stated -- a location, structure, "
    "measurement, or entity that A named AND the evidence supports -- which B no longer states? Answer ONLY "
    "about content the evidence SUPPORTS: if what A stated and B drops was WRONG / unsupported, that is NOT a "
    "loss (drops=false). If A stated an evidence-supported item (a specific structure, location, or entity) and "
    "B narrows to something else and OMITS it, that IS a loss (drops=true) -- the ideal answer keeps BOTH. Reply STRICT "
    "JSON: {{\"drops\": true|false, \"lost\": \"<the dropped supported slot, or empty>\"}}.\n\nA:\n{a}\n\n"
    "B:\n{b}\n\nVALIDATED EVIDENCE:\n{ev}\n")

def flip_drops_supported_content(a, b, evidence, judge_fn=None):
    """A core flip must not silently DROP an evidence-supported slot A already had. Example (MCTA-0): A 'portal
    vein thrombosis' -> B 'SMV thrombosis' narrows to SMV and drops portal, while the reference answer names
    both; the flip replaced instead of unioned. When B drops supported content A had, KEEP A (never below A).
    Only counts SUPPORTED drops: flipping away from a WRONG A (cartilage->myocardium) is not a loss.
    Fail-SAFE: no judge / parse-fail -> False (do not over-block)."""
    if not judge_fn or not str(a or "").strip() or not str(b or "").strip():
        return False
    try:
        d = _json_obj(judge_fn(_FLIPDROP_PROMPT.format(a=str(a)[:1100], b=str(b)[:1100],
                                                       ev=_format_new_evidence(evidence)[:2500])))
    except Exception:
        return False
    return _as_bool(d.get("drops")) if isinstance(d, dict) else False


def is_reinterpretation_ambiguity(a, b, new_evidence, judge_fn=None):
    """True iff B contradicts A merely by RE-INTERPRETING the SAME image finding (perceptual ambiguity) rather
    than by INDEPENDENT new evidence that resolves it. This is how correct->wrong flips arise: re-acquisition
    is not idempotent -- the VLM re-reads the same feature differently (thrombosis vs gas), the agent flips,
    and the selector would adopt the re-read as 'supporting' the flip. When the flip rests on ambiguity the
    harness cannot know which reading is right -> KEEP A. Fail-SAFE: no judge / parse-fail -> False (do not
    over-block)."""
    if not judge_fn or not str(a or "").strip() or not str(b or "").strip():
        return False
    try:
        d = _json_obj(judge_fn(_REINTERP_PROMPT.format(a=str(a)[:1200], b=str(b)[:1200],
                                                       ev=_format_new_evidence(new_evidence)[:2500])))
    except Exception:
        return False
    return _as_bool(d.get("ambiguous")) if isinstance(d, dict) else False


def is_hedge(a, b, judge_fn=None):
    """True iff B is a less-informative HEDGE of A (drops A's finding / retreats to uncertainty without adding
    a new one). Fail-SAFE: no judge / parse-fail -> False (do not over-block). Used to stop the recovery from
    adopting an uncertain-acquisition-driven hedge that DEGRADES a good answer."""
    if not judge_fn or not str(a or "").strip() or not str(b or "").strip():
        return False
    try:
        d = _json_obj(judge_fn(_HEDGE_PROMPT.format(a=str(a)[:1600], b=str(b)[:1600])))
    except Exception:
        return False
    return _as_bool(d.get("hedge")) if isinstance(d, dict) else False


def preserves_non_target_content(a, b, critique, judge_fn=None):
    """True iff B did not delete/contradict non-target content of A. Non-fatal: no judge / parse-fail -> True
    (the core-change gate is the primary protection; this is a secondary guard)."""
    if not judge_fn:
        return True
    prompt = _PRESERVE_PROMPT.format(critique=str(critique or "(none)")[:800], a=str(a or "")[:2000],
                                     b=str(b or "")[:2000])
    try:
        d = _json_obj(judge_fn(prompt))
    except Exception:
        return True
    if not isinstance(d, dict):
        return True
    return _as_bool(d.get("preserved"), default=True)


_AREFUTE_PROMPT = (
    "ORIGINAL answer A was already well-supported by the PRIOR evidence (E1). You are given ONLY the NEWLY "
    "acquired evidence (E2). Decide whether E2 provides an INDEPENDENT, DIRECT REFUTATION of A: a genuinely NEW "
    "observation -- a DIFFERENT region/attribute/measurement, or external reference knowledge -- that DIRECTLY "
    "contradicts A and would make a careful reader ABANDON A. It is NOT an independent refutation if E2 merely "
    "RE-INTERPRETS the same finding A was about (perceptual ambiguity), is vague/weakly suggestive, is general "
    "knowledge not tied to THIS specific image, or merely ADDS a detail without contradicting A. Reply STRICT "
    "JSON: {{\"refutes\": true|false, \"why\": \"<short>\"}}. Default refutes=FALSE if uncertain.\n\n"
    "A:\n{a}\n\nNEWLY ACQUIRED EVIDENCE (E2 only):\n{ev}\n")

def new_evidence_independently_refutes(a, new_evidence, judge_fn=None):
    """Does the NEW evidence (E2 ONLY) INDEPENDENTLY and DIRECTLY refute A? Used to gate a flip AWAY from a
    ROBUST A: a good answer must not be overturned by a same-finding re-read, a vague look, general knowledge,
    or a mere addition. Fail-CLOSED: no judge / no new evidence / parse-fail / uncertain -> False (PROTECT A)."""
    if not judge_fn or not str(a or "").strip() or not new_evidence:
        return False
    try:
        d = _json_obj(judge_fn(_AREFUTE_PROMPT.format(a=str(a)[:1100], ev=_format_new_evidence(new_evidence)[:2500])))
    except Exception:
        return False
    return _as_bool(d.get("refutes")) if isinstance(d, dict) else False


_SLOT_COMPLETE_PROMPT = (
    "A medical visual-QA task. The PUBLIC QUESTION can require MULTIPLE answer SLOTS (e.g. 'identify the tissue "
    "AND determine the stain' -> slots: tissue, stain; 'diagnosis and subtype'; 'value and the ratio'; a "
    "two-part comparison). The ORIGINAL answer A and a REVISED answer B share the SAME primary conclusion (no "
    "core flip). Decide whether B should be adopted because it COMPLETES a required slot that A left MISSING or "
    "vague, WITH NO regression. Using the VALIDATED EVIDENCE, reply adopt=true ONLY IF ALL FOUR hold:\n"
    "(1) PRESERVE: B keeps every slot A already answered correctly -- nothing A stated is dropped, weakened, "
    "hedged, or contradicted (B is a SUPERSET of A content);\n"
    "(2) COMPLETE: B fills at least one slot the PUBLIC QUESTION requires but A left missing/vague;\n"
    "(3) SUPPORTED: that newly-filled slot is DIRECTLY supported by the validated evidence, not guessed;\n"
    "(4) CLEAN: B adds NO new unsupported claim and changes no other slot.\n"
    "If B merely rephrases, hedges, drops or softens detail, or the added slot is unsupported/not required -> "
    "adopt=false. Reply STRICT JSON: {{\"adopt\": true|false, \"slot\": \"<the completed slot or empty>\", "
    "\"why\": \"<short>\"}}.\n\nPUBLIC QUESTION:\n{goal}\n\nORIGINAL A:\n{a}\n\nREVISED B:\n{b}\n\n"
    "VALIDATED EVIDENCE:\n{ev}\n")

def slot_completion_check(goal, pub, a, b, evidence, judge_fn=None):
    """A NON-CORE B (same primary conclusion) may still ADD a required answer slot A missed (stain, subtype,
    measurement, the second half of a two-part question). Under a lenient contains-reference answer score this
    can only raise the score. Adopt B ONLY when it is a validated SUPERSET of A: preserves every slot A
    answered, fills a PUBLIC-QUESTION-required slot A missed, that slot is evidence-supported, and no
    unsupported claim is introduced. Delivering such a B == 'A + validated slot patch' with NO loss. Fail-SAFE:
    no judge / parse-fail / missing text -> {adopt: False} (fall back to KEEP A)."""
    if not judge_fn or not str(a or "").strip() or not str(b or "").strip():
        return {"adopt": False}
    ev = _format_new_evidence(evidence) or "(none)"
    try:
        d = _json_obj(judge_fn(_SLOT_COMPLETE_PROMPT.format(goal=str(goal or "")[:900], a=str(a)[:1100],
                                                            b=str(b)[:1100], ev=ev[:2600])))
    except Exception:
        return {"adopt": False}
    if not isinstance(d, dict):
        return {"adopt": False}
    return {"adopt": _as_bool(d.get("adopt")), "slot": str(d.get("slot") or "").strip()[:40]}


_MERGE_PROMPT = (
    "Produce the FINAL answer to the PUBLIC QUESTION by MERGING the original answer A and the revised answer B "
    "at the CLAIM level, using ONLY the VALIDATED EVIDENCE -- deliver A + a validated patch (a UNION), never a "
    "wholesale replacement. RULES: (1) KEEP every claim in A that the evidence SUPPORTS and that is not "
    "explicitly contradicted -- preserve A wording for those; (2) ADD a claim from B ONLY if it is DIRECTLY "
    "supported by the evidence, relevant to the question, and not already in A; (3) DROP a claim from A ONLY if "
    "the evidence EXPLICITLY contradicts it; (4) add NOTHING the evidence does not support -- no new diagnoses, "
    "locations, values, or hedged claims. Output ONLY the merged answer text as a concise clinical answer, no "
    "explanation, no JSON.\n\nPUBLIC QUESTION:\n{goal}\n\nA:\n{a}\n\nB:\n{b}\n\nVALIDATED EVIDENCE:\n{ev}\n")

def build_merged_answer(a, b, goal_spec, all_evidence, new_evidence, judge_fn=None):
    """CLAIM-LEVEL MERGE: deliver A + validated patch, not a wholesale B. Keeps A's supported claims
    (verbatim), adds only B claims the evidence directly supports, drops only A claims the evidence
    explicitly contradicts, adds nothing unsupported. This structurally removes the 'B silently
    rewrites/drops A' risk for BOTH slot completion and flips. Fail-safe: no judge / empty output -> None
    (caller falls back to the reverified B)."""
    if not judge_fn or not str(a or "").strip() or not str(b or "").strip():
        return None
    goal = goal_spec.get("goal") if isinstance(goal_spec, dict) else goal_spec
    ev = _format_new_evidence(all_evidence) or _format_new_evidence(new_evidence) or "(none)"
    try:
        out = judge_fn(_MERGE_PROMPT.format(goal=str(goal or "")[:800], a=str(a)[:1100], b=str(b)[:1100], ev=ev[:2800]))
    except Exception:
        return None
    out = (out or "").strip().strip("`").strip()
    if out.lower().startswith("answer:"):
        out = out[7:].strip()
    return out or None


def evaluate_candidate(intervention, candidate_b, goal_spec, all_evidence, new_evidence,
                       judge_fn=None, reverify_fn=None, allow_type="ALLOW"):
    """UNIFIED answer-surface adoption gate for inline/external REVISE and ACQUIRE/CONSULT re-answers.
    Returns (decision, reason_code) where decision in {"ADOPT_B","KEEP_A"}; reason_code IS the disposition.
    Ordered, fail-safe gate:
      1. core-change gate: a core flip needs new validated evidence that DIRECTLY supports it (delta only);
      2. public-dimension comparator must clearly prefer B;
      3. B must not introduce a new unsupported claim (covered by adopt_revised's revised_new_hard_violation)
         and must preserve non-target content;
      4. B must pass final re-verification (before_final ALLOW), else rollback.
    Anything uncertain -> KEEP_A."""
    a = intervention.get("root_answer", "")
    itype = intervention.get("type", "REVISE")
    _ans = intervention.get("surface") == "answer"   # SURFACE GATE: the MedCTA answer-surface additions below
    # (A-quality gate, ambiguity/flip-drop guards, slot completion) apply ONLY to the answer surface. An
    # intervention without the surface flag skips them.
    critique = intervention.get("critique")
    has_new = itype in ("ACQUIRE", "CONSULT") and bool(new_evidence)
    goal = goal_spec.get("goal") if isinstance(goal_spec, dict) else goal_spec
    pub = goal_spec.get("public_context") if isinstance(goal_spec, dict) else None

    sig_a = extract_decision_signature(goal_spec, a, judge_fn)
    sig_b = extract_decision_signature(goal_spec, candidate_b, judge_fn)
    core = compare_decision_signatures(sig_a, sig_b, judge_fn)
    core_changed = bool(core.get("changed"))
    _delta_ok = False
    _strong_delta = False
    if core_changed:
        if not has_new:
            return ("KEEP_A", "kept_original_core_flip_without_new_evidence")
        # A-QUALITY GATE: assess A on the PRE-INTERVENTION evidence ONLY (E1), NEVER the pool that includes the
        # new E2 -- otherwise a bad re-read weakens A's own protection. A ROBUST A (already supported by E1) may be
        # flipped ONLY by an INDEPENDENT, DIRECT refutation from the new evidence; a same-finding re-read / vague
        # look / general knowledge / mere addition does NOT qualify. Fail-CLOSED: if A is robust and the new
        # evidence does not independently refute it -> KEEP A. Example (MCTA-0): A = portal thrombosis is robust on
        # E1; a gas re-read, an added SMV finding or an added splenic finding does not refute it.
        _baseline_ev = None
        if _ans:
            _new_ids = {e.get("evidence_id") for e in (new_evidence or [])}
            _base_ids = intervention.get("baseline_ev_ids") or set()
            _baseline_ev = [e for e in all_evidence if e.get("evidence_id") in _base_ids and e.get("evidence_id") not in _new_ids]
            if not _baseline_ev:
                _baseline_ev = [e for e in all_evidence if e.get("evidence_id") not in _new_ids]
            _a_base = verify_claim_support(goal, pub, a, _baseline_ev, judge_fn)
            _a_robust = getattr(_a_base, "supported", None) is True and (getattr(_a_base, "confidence", 0.0) or 0.0) >= 0.6
            if _a_robust and not new_evidence_independently_refutes(a, new_evidence, judge_fn):
                return ("KEEP_A", "kept_original_A_robust_not_independently_refuted")
        # TWO-STAGE support, not "E2 ALONE proves the full flip". A MedCTA answer usually needs E1+E2 joint
        # reasoning; a valid E2 is often only a discriminating INCREMENT.
        #   STAGE 1 -- delta CONTRIBUTION: did the newly acquired evidence causally contribute to the change
        #             (supported + names supporting evidence ids)? This is relevance, not full proof.
        #   STAGE 2 -- TOTAL post-intervention support: does the FULL validated evidence set (E1 + E2 + prior)
        #             support B? A core flip is legitimate when ALL the evidence supports it even if E2 alone
        #             is below a strict single-delta bar. Oracle-blind (judged vs delivered evidence, not gold);
        #             the comparator + before_final reverify below remain the T->F guards, so this is NOT a
        #             loosened safety gate -- it fixes the EVIDENCE SET the support judge is allowed to read.
        support = verify_change_against_delta(goal, sig_a, sig_b, new_evidence, judge_fn)
        _delta_relevant = bool(_as_bool(support.get("supported")) and support.get("supporting_evidence_ids"))
        _strong_delta = _delta_relevant and (support.get("confidence") or 0.0) >= 0.8
        _total = verify_claim_support(goal, pub, candidate_b, all_evidence, judge_fn)
        _total_ok = getattr(_total, "supported", None) is True
        if not (_strong_delta or (_delta_relevant and _total_ok)):
            return ("KEEP_A", "kept_original_core_change_not_supported_by_delta")
        _delta_ok = True   # core change is evidence-grounded (strong single delta OR relevant delta + total support)
        # AMBIGUITY GUARD (correct->wrong flips): re-acquisition is NOT idempotent -- the VLM re-reads the SAME
        # image finding differently (thrombosis<->gas, cartilage<->muscle), the agent flips, and the "evidence"
        # that supports B is just that contradictory RE-READ, not independent new information. Being oracle-blind,
        # the harness cannot know which reading is right, so a flip resting on a same-finding re-interpretation
        # AMBIGUITY must KEEP A; otherwise the harness can turn a correct answer wrong.
        if _ans and is_reinterpretation_ambiguity(a, candidate_b, new_evidence, judge_fn):
            return ("KEEP_A", "kept_original_reinterpretation_ambiguity")
        # PRESERVE guard on FLIPS (the preserve rule applied to core changes too): a flip must not silently DROP
        # an evidence-supported slot A already had. Example (MCTA-0): A 'portal vein thrombosis' -> B narrowed to
        # 'SMV thrombosis' and dropped portal, while the reference answer names both; the flip replaced instead of
        # unioned. Dropping SUPPORTED content -> KEEP A (never deliver below A).
        # Use BASELINE evidence (E1), not the pool that includes E2 -- a WRONG re-observation must not erase A's
        # own supported content (e.g. a splenic re-localisation making 'portal' look unsupported).
        _bev = _baseline_ev or all_evidence
        if _ans and flip_drops_supported_content(a, candidate_b, _bev, judge_fn):
            return ("KEEP_A", "kept_original_flip_drops_supported_slot")

    # NON-CORE RETENTION (the recovery must NOT break the outcome). An ACQUIRE/REVISE re-answer with the SAME
    # core tends to drop specificity / hedge / rephrase worse, and a single stochastic hedge-judge cannot
    # reliably catch it. The recovery's legitimate value is a CONFIDENT CORE FLIP (wrong core -> right core,
    # evidence-supported). A non-core refinement therefore KEEPS A unconditionally -> the harness never harms a
    # good answer and never rewrites a same-core answer on uncertain evidence. (Adding specificity that truly
    # matters registers as a CORE change and flows through the evidence-grounded path above.)
    if not core_changed:
        # SLOT-COMPLETION ADOPTION -- ANSWER SURFACE ONLY. Complete a required answer slot A left missing (stain,
        # subtype, measurement, 2nd half of a two-part question) as a validated SUPERSET of A. Without the surface
        # flag a non-core B is always kept as A.
        if _ans:
            _sc = slot_completion_check(goal, pub, a, candidate_b, (new_evidence or all_evidence), judge_fn)
            if _sc.get("adopt") and not is_hedge(a, candidate_b, judge_fn):
                return ("ADOPT_B", "adopted_slot_completion_%s" % (_sc.get("slot").replace(" ", "_") or "slot"))
        return ("KEEP_A", "kept_original_noncore_refinement_not_adopted")

    cmp = compare_answer_candidates(goal, pub, a, candidate_b, critique, all_evidence, judge_fn=judge_fn)
    if cmp.get("preferred") == "uncertain":
        return ("KEEP_A", "kept_original_comparator_uncertain")
    _adopt = adopt_revised(cmp)                       # strict: preferred==revised AND margin>=tau AND resolved
    # EVIDENCE-GROUNDED SMALL-MARGIN ADOPTION (raises P(F->T | engaged) without blindly loosening safety).
    # The comparator LEANS to B (preferred==revised, critique resolved, no new hard violation) but is below
    # the strict margin. Adopt B anyway ONLY when the change is POSITIVELY evidence-grounded:
    #   (a) a core change that already passed the >=0.8 delta gate with supporting evidence ids, OR
    #   (b) a non-core refinement whose B is directly supported by the freshly-acquired evidence.
    # reverify (below) still guards; never fires on preferred==uncertain/original or a new hard violation.
    if (not _adopt and cmp.get("preferred") == "revised"
            and cmp.get("critique_resolved") and not cmp.get("revised_new_hard_violation")):
        if _delta_ok:
            _adopt = True
        elif has_new:
            _es = verify_claim_support(goal, pub, candidate_b, new_evidence, judge_fn)
            if getattr(_es, "supported", None) is True:
                _adopt = True
    if not _adopt:                                    # incl. not revised_new_hard_violation (no new unsupported claim)
        return ("KEEP_A", "kept_original_candidate_not_clearly_better")
    # A scoped REVISE must preserve non-target content; but an ACQUIRE re-answer legitimately RE-DERIVES the
    # WHOLE answer from newly acquired evidence -> the entire answer IS the target, so the non-target-content
    # preservation gate does not apply (T->F still guarded by the comparator + evidence-support above).
    if itype not in ("ACQUIRE", "CONSULT") and not preserves_non_target_content(a, candidate_b, critique, judge_fn):
        return ("KEEP_A", "kept_original_non_target_mutation")

    # ANTI-HEDGE gate: uncertain acquisitions can drive the agent to DROP a correct specific finding and
    # retreat to "no clear difference", which would otherwise pass as a non-core refinement. Never adopt a B
    # that is LESS informative than A unless a CONFIDENT evidence-supported CORE flip justifies rewriting the
    # whole answer. A hedge with only uncertain/negative new evidence -> KEEP A.
    _confident_core_flip = bool(core_changed and _strong_delta) if core_changed else False
    if not _confident_core_flip and is_hedge(a, candidate_b, judge_fn):
        return ("KEEP_A", "kept_original_candidate_is_hedge")

    if reverify_fn is not None:
        try:
            rv = reverify_fn(candidate_b)
        except Exception:
            rv = None
        if not (rv is not None and getattr(rv, "type", None) == allow_type):
            return ("KEEP_A", "rolled_back_candidate_reverify_failed")

    return ("ADOPT_B", "adopted_evidence_supported_core_change")


# =====================================================================================================
# THE HOOKS. after_final_verdict opens the hold; on_answer consumes the next final as candidate B and returns
# Read (another round of observation), Repair (a re-answer round) or Accept (the delivered text).
# =====================================================================================================
_REDERIVE_TEXT = ("Your prior answer was not grounded in a direct observation of the "
                  "image. A targeted observation of %s (%s) has now been performed for "
                  "you; its full result is in your last observation. Your prior answer "
                  "was: \"%s\". Re-derive your answer to the question PRIMARILY from "
                  "this new observation -- do NOT simply restate the prior answer. State "
                  "the specific finding the observation shows and the conclusion it "
                  "supports.%s")


class AnswerCandidate(object):
    """The open answer-surface intervention of one task, from the kernel verdict that opens it to the delivered text."""
    name = "answer_candidate"

    def __init__(self, task):
        # root_answer (A), baseline_ev_ids, type (ACQUIRE / REVISE), critique, acquisition_goal, acq_round, rev_round
        self.pending = None

    @staticmethod
    def _root(ctx, answer):
        """Root A: the agent's EARLIEST answer that reached the attempt point, else the one under review. Protects
        the earliest answer from later harness-driven drift."""
        answers = ctx.answers
        return (answers[0].text if answers else "") or answer

    def after_final_verdict(self, ctx, answer, verdict):
        """The kernel's before_final did not ALLOW `answer`. Open the hold for an ACQUIRE or a candidate REVISE.

        This runs before the shell plans the read and before it checks the repair cap. The two outcomes the hook
        cannot see both end the episode (a refused read commits the final clean; cap exhaustion delivers
        kept_original_no_candidate), so opening the hold here is not observable. -> the agent text of an ACQUIRE."""
        extra = ((getattr(verdict, "raw", None) and getattr(verdict.raw, "extra", None)) or {})
        if verdict.type == "ACQUIRE":
            aq = next_action_of(verdict).get("args") or {}
            hyps = extra.get("hypotheses") or aq.get("hypotheses") or []
            hyp_txt = ((" Distinguish specifically between: %s."
                        % "; ".join(str(h) for h in hyps[:4])) if hyps else "")
            region = aq.get("region"); attribute = aq.get("attribute")
            # ACQUIRE critique = the acquisition OBJECTIVE (region/attribute/hypotheses), so the comparator can judge
            # whether B actually resolved the targeted observation (critique=None could never set critique_resolved).
            critique = ("Resolve the missing direct observation of %s (%s)%s and answer the "
                        "target feature from it." % (region, attribute,
                        (" -- distinguish %s" % "; ".join(str(h) for h in hyps[:4])) if hyps else ""))
            # PRE-acquisition baseline -> post-acquire B judged vs the evidence delta
            self.pending = {"surface": "answer", "type": "ACQUIRE",
                            "root_answer": self._root(ctx, answer),
                            "baseline_ev_ids": {e.get("evidence_id") for e in ctx.ledger.evidence},
                            "critique": critique,
                            "acquisition_goal": {"region": region, "attribute": attribute,
                                                 "hypotheses": list(hyps)[:4]}}
            # STRUCTURED RE-ENTRY: the prior answer was NOT grounded in an actual observation. Give A + the specific
            # gap (region/attribute) + hypotheses, and REQUIRE re-derivation from the new observation -- never bias
            # toward keeping A ("change only if contradicts" under-adopts a better evidence-grounded answer).
            return _REDERIVE_TEXT % (aq.get("region"), aq.get("attribute"), str(answer)[:400], hyp_txt)
        if verdict.type in ("REVISE", "BLOCK") and extra.get("candidate"):   # keep A, ask for a revised B
            self.pending = {"surface": "answer", "type": "REVISE",
                            "root_answer": self._root(ctx, answer), "critique": extra.get("critique"),
                            "baseline_ev_ids": {e.get("evidence_id") for e in ctx.ledger.evidence}}
        return None

    def on_answer(self, ctx, answer):
        """Consume this final as candidate B for the open intervention; None when nothing is held."""
        pending_iv = self.pending
        if pending_iv is None:
            return None
        _A = pending_iv.get("root_answer", ""); _B = answer
        _cm = ctx.contract_meta or {}
        _base_ids = pending_iv.get("baseline_ev_ids") or set()
        # evidence DELTA = validated, non-foreign evidence acquired AFTER the intervention started
        _new_ev = [e for e in ctx.ledger.evidence if e.get("evidence_id") not in _base_ids
                   and e.get("status") == "VALIDATED" and e.get("scope_relation") != "foreign"]
        _acq_round = pending_iv.get("acq_round", 1)
        # Acquisition rounds (real perception reads) and answer-revision rounds (text-only re-answers) are
        # SEPARATE budgets. A no-information text re-answer must NOT consume an acquisition round.
        _rev_round = pending_iv.get("rev_round", 0)
        # SINGLE before_final(B): before_final consumes the semantic budget, so it is called EXACTLY ONCE here
        # and its result reused. EVIDENCE SUFFICIENCY BEFORE THE SELECTOR: a residual coverage gap drives
        # ANOTHER acquisition round -- independent of candidate adoption.
        try:
            _rvB = ctx.verify_final(_B)            # its events are the ones the trajectory keeps (ANY type)
        except Exception as _re:
            _rvB = None; ctx.error("reverify_B: %r" % _re)
        _rvB_type = getattr(_rvB, "type", "ALLOW") if _rvB is not None else "ALLOW"
        _rvB_na = ((getattr(_rvB, "raw", None) and getattr(_rvB.raw, "extra", None)) or {}).get("next_action") or {}
        _rvB_extra = ((getattr(_rvB, "raw", None) and getattr(_rvB.raw, "extra", None)) or {})
        # ROUND N+1 (bounded): B still has a residual gap. A ROUTABLE ACQUIRE -> fetch another observation; a
        # REVISE/BLOCK (or ACQUIRE with no routable tool) -> let the agent RE-ANSWER (B2) with the harness critique.
        if _acq_round < ctx.semantic_rounds("acquire") and _rvB_type == "ACQUIRE" and pending_iv.get("type") == "ACQUIRE" \
                and _rvB_na.get("tool") and _rvB_na.get("read_only"):
            _prior_tool = (pending_iv.get("acquisition_goal") or {}).get("tool")
            _aq2 = _rvB_na.get("args") or {}
            _hy2 = _rvB_extra.get("hypotheses") or _aq2.get("hypotheses") or []
            _feedback = {"decision": "ACQUIRE", "stage": "before_final",
                "reason": ("A further targeted observation of %s (%s) has now been performed; re-derive "
                           "your answer from ALL observations gathered so far."
                           % (_aq2.get("region"), _aq2.get("attribute"))), "missing_obligations": []}
            _na = next_action_of(_rvB)
            _read = Read(_na.get("tool"), _na.get("args"), feedback=_feedback, source=_rvB)
            _dispatchable = ctx.plan_read(_read)   # False iff the shared actuator refused (budget / malformed)
            pending_iv["acq_round"] = _acq_round + 1
            pending_iv["acquisition_goal"] = {"region": _aq2.get("region"), "attribute": _aq2.get("attribute"),
                                              "hypotheses": list(_hy2)[:4], "tool": _rvB_na.get("tool")}
            pending_iv["critique"] = ("Resolve the still-missing observation of %s (%s); answer the target "
                                      "feature from ALL observations gathered so far."
                                      % (_aq2.get("region"), _aq2.get("attribute")))
            ctx.emit({"event_type": "adaptive_acquisition", "round": _acq_round + 1,
                      "region": _aq2.get("region"), "attribute": _aq2.get("attribute"),
                      "tool": _rvB_na.get("tool"),
                      "same_tool_as_prior": bool(_prior_tool) and _rvB_na.get("tool") == _prior_tool,
                      "status": "ok"})
            if _dispatchable:                      # a refused acquisition is not a completed round
                return _read                       # round-(N+1) runs, agent re-answers B(N+1)
        elif _rev_round < ctx.semantic_rounds("revise") and _rvB_type in ("REVISE", "BLOCK", "ACQUIRE"):
            # residual grounding/content gap (or an ACQUIRE the harness could not route) -> the agent RE-ANSWERS
            # B2 with the harness critique instead of B being rolled back to A. Consumes a REVISION round.
            _feedback = ctx.verdict_feedback(_rvB, "before_final")
            pending_iv["rev_round"] = _rev_round + 1
            if _rvB_extra.get("critique"):
                pending_iv["critique"] = _rvB_extra.get("critique")
            ctx.emit({"event_type": "adaptive_revise", "round": _rev_round + 1, "rvB_type": _rvB_type,
                      "status": "ok"})
            return Repair(_feedback)
        # evidence sufficient (ALLOW) or acquisition budget exhausted -> judge A vs B. Reuse the single _rvB.
        try:
            # Pass one candidate_evidence_view (VALIDATED, non-foreign) so stage-2 total-support and the comparator
            # judge the SAME evidence set the final reverify filters to -- not the raw ledger.
            _cand_ev_view = [e for e in ctx.ledger.evidence
                             if e.get("status") == "VALIDATED" and e.get("scope_relation") != "foreign"]
            _decision, _disp = evaluate_candidate(
                pending_iv, _B, _cm, _cand_ev_view, _new_ev,
                judge_fn=ctx.semantic_check(),
                reverify_fn=(lambda _ans, _pre=_rvB: _pre))
        except Exception as _re:
            ctx.error("evaluate_candidate: %r" % _re)
            _decision, _disp = "KEEP_A", "kept_original_gate_error"
        _flag = None
        if _decision == "ADOPT_B":
            # CLAIM-LEVEL MERGE: deliver A + validated patch (a UNION), not a wholesale B that could silently
            # rewrite/drop A supported content. Build the merge, REVERIFY it; deliver it only if it passes, else fall
            # back to B (already reverified) -> never regress the gate.
            _chosen = _B; _rv_clean = True
            try:
                _merged = build_merged_answer(_A, _B, _cm, _cand_ev_view, _new_ev, ctx.semantic_check())
            except Exception as _me:
                ctx.error("merge: %r" % _me); _merged = None
            if _merged and _merged.strip() and _merged.strip() != (_B or "").strip():
                _rvm = None
                try: _rvm = ctx.verify_final(_merged)
                except Exception as _me: ctx.error("merge reverify: %r" % _me)
                if _rvm is not None and getattr(_rvm, "type", None) == "ALLOW":
                    _chosen = _merged; _disp = _disp + "+claimmerge"
        else:
            _chosen = _A   # keep root A; commit it (flagged if not itself clean)
            _rv = None
            try: _rv = ctx.verify_final(_chosen)
            except Exception as _re: ctx.error("keepA re-verify: %r" % _re)
            _rv_clean = (_rv is not None and _rv.type == "ALLOW")
            if not _rv_clean:
                _flag = _disp
        self.pending = None
        return Accept(_chosen, _disp, pending_iv.get("type"), _rv_clean, _flag)
