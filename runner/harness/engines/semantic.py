"""Semantic engine — judge-backed checks the deterministic engine cannot do (claim<->evidence support).

The judge is INJECTED as a `judge_fn(prompt:str) -> str|None` (the kernel builds it from the gateway with
an INDEPENDENT judge model, budgeted by max_semantic_checks). This keeps capabilities model-agnostic and
unit-testable (tests inject a fake judge). Fail-SAFE: no judge -> UNKNOWN (never fabricate a pass), and a
low-confidence verdict -> caller ESCALATEs rather than silently allowing.
"""
import json

UNKNOWN = "UNKNOWN"

# Relation tri-state (P0-5): distinguish a real CONTRADICTION (answer conflicts with the evidence) from
# under-coverage (INSUFFICIENT: the evidence neither supports nor refutes). The conservative gate uses
# this to hard-REVISE only on contradiction and to gracefully degrade on under-coverage.
SUPPORTED = "supported"
CONTRADICTED = "contradicted"
INSUFFICIENT = "insufficient"


class SemanticVerdict:
    __slots__ = ("supported", "confidence", "reason", "relation", "critical_claim", "evidence_ids")

    def __init__(self, supported=None, confidence=0.0, reason=None, relation=None,
                 critical_claim=None, evidence_ids=None):
        self.supported = supported       # True / False / None(unknown)
        self.confidence = confidence     # 0..1
        self.reason = reason
        self.relation = relation         # supported | contradicted | insufficient | None
        self.critical_claim = critical_claim   # the ONE specific answer claim the evidence refutes (localizes a contradiction)
        self.evidence_ids = list(evidence_ids or [])   # which numbered evidence items refute it (E1, E2, ...)

    def localizable(self):
        """A contradiction is MUST-RESOLVE-eligible only if it names a specific claim AND cites evidence."""
        return bool(self.critical_claim and str(self.critical_claim).strip()) and bool(self.evidence_ids)

    def to_dict(self):
        return {"supported": self.supported, "confidence": round(float(self.confidence or 0.0), 3),
                "reason": self.reason, "relation": self.relation,
                "critical_claim": self.critical_claim, "evidence_ids": self.evidence_ids}


_PROMPT = (
    "You are an evidence auditor. Using the QUESTION/GOAL and the SELECTED evidence below, decide ONLY "
    "whether the agent's FINAL ANSWER is SUPPORTED by THIS evidence — do NOT judge whether the answer is "
    "correct in general and do NOT use outside knowledge. Classify the relation as exactly one of: "
    "'supported' (evidence directly supports the answer), 'contradicted' (evidence conflicts with/refutes "
    "the answer), or 'insufficient' (evidence neither supports nor contradicts — it under-covers the answer). "
    "Reply with strict JSON: "
    '{{"relation": "supported|contradicted|insufficient", "confidence": 0.0-1.0, "reason": "<short>", '
    '"critical_claim": "<for contradicted ONLY: the single specific claim in the ANSWER that THIS evidence '
    'refutes; else null>", "evidence_ids": ["<for contradicted ONLY: the evidence tags e.g. E1, E2 that '
    'refute it; else empty>"]}}.\n\n'
    "QUESTION / GOAL:\n{task_goal}\n\nPUBLIC TASK CONTEXT:\n{public_context}\n\n"
    "SELECTED EVIDENCE:\n{evidence}\n\nFINAL ANSWER:\n{answer}\n"
)


def verify_claim_support(task_goal, public_context, answer, evidence, judge_fn=None):
    """Is `answer` supported by the SELECTED `evidence` (already filtered by the postcondition's evidence
    selector — this function does NOT see unrelated evidence). `task_goal` and `public_context` are the
    PUBLIC task info the agent already sees (compiler whitelist, leak-checked) — NEVER gold_answer/reference;
    they only tell the judge WHAT the answer is claiming. Returns a SemanticVerdict. No judge -> UNKNOWN."""
    if not judge_fn:
        return SemanticVerdict(None, 0.0, "semantic_judge_unavailable", relation=None)
    ev_text = _format_evidence(evidence)
    if not ev_text.strip():
        # zero SELECTED evidence = under-coverage (INSUFFICIENT), NOT a high-confidence contradiction.
        return SemanticVerdict(None, 1.0, "no_selected_evidence", relation=INSUFFICIENT)
    prompt = _PROMPT.format(task_goal=str(task_goal or "(not provided)")[:2000],
                            public_context=str(public_context or "(none)")[:3000],
                            evidence=ev_text[:8000], answer=str(answer)[:2000])
    try:
        raw = judge_fn(prompt)
    except Exception as ex:
        return SemanticVerdict(None, 0.0, "semantic_judge_error:%r" % (ex,))
    return _parse(raw)


def _format_evidence(evidence):
    out = []
    for e in (evidence or []):
        if isinstance(e, dict):
            out.append("[E%d] [%s] %s" % (len(out) + 1, e.get("type", "evidence"),
                                           str(e.get("value_full") or e.get("value", ""))[:1800]))
        else:
            out.append("[E%d] %s" % (len(out) + 1, str(e)[:300]))
    return "\n".join(out)


def _parse(raw):
    if not raw:
        return SemanticVerdict(None, 0.0, "empty_judge_response", relation=None)
    s = raw if isinstance(raw, str) else str(raw)
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        try:
            d = json.loads(s[i:j + 1])
            conf = float(d.get("confidence", 0.0) or 0.0)
            reason = d.get("reason")
            rel = d.get("relation")
            rel = rel.strip().lower() if isinstance(rel, str) else None
            if rel not in (SUPPORTED, CONTRADICTED, INSUFFICIENT):
                rel = None
            sup = d.get("supported")
            # STRICT: only an actual JSON boolean counts. A string "false" must NOT become True.
            supported = sup if isinstance(sup, bool) else None
            # Reconcile relation <-> supported. Back-compat: a {"supported": bool} judge carries no
            # relation -> derive it (true->supported, false->contradicted). A {"relation": ...} judge
            # carries no boolean -> derive supported (insufficient -> None, never a contradiction).
            if rel is None and supported is not None:
                rel = SUPPORTED if supported else CONTRADICTED
            elif rel is not None and supported is None:
                supported = True if rel == SUPPORTED else (False if rel == CONTRADICTED else None)
            cc = d.get("critical_claim")
            cc = cc.strip() if isinstance(cc, str) and cc.strip().lower() not in ("", "null", "none") else None
            eids = d.get("evidence_ids")
            eids = [str(x) for x in eids if str(x).strip()] if isinstance(eids, list) else []
            return SemanticVerdict(supported, conf, reason, relation=rel, critical_claim=cc, evidence_ids=eids)
        except Exception:
            pass
    return SemanticVerdict(None, 0.0, "unparseable_judge_response", relation=None)


# ---------------------------------------------------------------------------------------------------------
# ADEQUACY AUDIT (Selective Epistemic Repair foundation). Unlike verify_claim_support (which only asks "is
# the answer supported by this evidence"), this audits whether the submitted answer ADEQUATELY ADDRESSES the
# PUBLIC task using the agent's validated evidence, and localizes specific evidence-linked defects, split
# into HARD violations (-> Layer-1 must-resolve) and REPAIRABLE gaps (-> Layer-2 candidate repair). It still
# never reads gold / infers a hidden reference answer.
# ---------------------------------------------------------------------------------------------------------
class SemanticAudit:
    __slots__ = ("addresses_task", "hard_violations", "repairable_gaps", "confidence", "reason")

    def __init__(self, addresses_task=None, hard_violations=None, repairable_gaps=None, confidence=0.0, reason=None):
        self.addresses_task = addresses_task
        self.hard_violations = list(hard_violations or [])
        self.repairable_gaps = list(repairable_gaps or [])
        self.confidence = confidence
        self.reason = reason

    def top_hard(self):
        """The first LOCALIZED hard violation (names a claim AND cites evidence) -> must-resolve eligible."""
        for h in self.hard_violations:
            if isinstance(h, dict) and str(h.get("claim") or "").strip() and (h.get("evidence_ids") or []):
                return h
        return None

    def top_gap(self):
        """The first LOCALIZED repairable gap (names a claim/critique) -> candidate-repair eligible."""
        for g in self.repairable_gaps:
            if isinstance(g, dict) and (str(g.get("critique") or "").strip() or str(g.get("claim") or "").strip()):
                return g
        return None

    def to_dict(self):
        return {"addresses_task": self.addresses_task, "hard_violations": self.hard_violations,
                "repairable_gaps": self.repairable_gaps, "confidence": round(float(self.confidence or 0.0), 3),
                "reason": self.reason}


_AUDIT_PROMPT = (
    "You are a clinical answer auditor. You are given a public QUESTION/GOAL, public task context, the agent's "
    "SELECTED VALIDATED evidence (numbered [E1]..), and the agent's submitted FINAL ANSWER. Do NOT "
    "independently solve the task or infer a hidden reference answer. Assess whether the submitted answer "
    "ADEQUATELY ADDRESSES THE PUBLIC TASK using the agent's validated evidence, and identify specific "
    "evidence-linked reasoning defects. Classify each defect as exactly one of:\n"
    "- HARD violation: the answer directly CONTRADICTS the evidence, claims an action succeeded that the "
    "evidence shows failed/unknown, or concerns the WRONG subject.\n"
    "- REPAIRABLE gap: the answer does not directly ANSWER the question; asserts MORE specificity than the "
    "evidence supports; IGNORES a decisive discriminating feature present in the evidence; has a clear GAP in "
    "the evidence->conclusion chain; or does not match the required OUTPUT FORM.\n"
    "Name the specific answer claim and cite the evidence ids for each defect. Reply with STRICT JSON:\n"
    '{{"addresses_task": true|false, "hard_violations": [{{"type": "contradiction|false_success|wrong_subject", '
    '"claim": "<answer claim>", "evidence_ids": ["E1"], "reason": "<short>"}}], "repairable_gaps": '
    '[{{"type": "unanswered|over_specific|missed_discriminator|reasoning_gap|form_mismatch", "claim": '
    '"<answer claim>", "evidence_ids": ["E1"], "critique": "<specific, evidence-linked>"}}], '
    '"confidence": 0.0-1.0}}.\n\n'
    "QUESTION / GOAL:\n{task_goal}\n\nPUBLIC TASK CONTEXT:\n{public_context}\n\n"
    "SELECTED EVIDENCE:\n{evidence}\n\nFINAL ANSWER:\n{answer}\n"
)


def audit_answer(task_goal, public_context, answer, evidence, judge_fn=None, goal_spec=None):
    """Adequacy audit -> SemanticAudit{addresses_task, hard_violations[], repairable_gaps[], confidence}.
    No judge -> empty audit (fail-safe: no fabricated defects). Evidence is the SAME selector-filtered,
    VALIDATED set the claim-support check sees."""
    if not judge_fn:
        return SemanticAudit(None, [], [], 0.0, "audit_judge_unavailable")
    ev_text = _format_evidence(evidence)
    _gs = ("\n\nThe task EXPLICITLY requires (structured): " + json.dumps(goal_spec, ensure_ascii=False)
           + "\nAlso flag, as a REPAIRABLE gap, any required_effect/required_field/requested_operation the "
             "answer does not satisfy.") if goal_spec else ""
    prompt = _AUDIT_PROMPT.format(task_goal=str(task_goal or "(not provided)")[:2000],
                                  public_context=(str(public_context or "(none)")[:3000] + _gs),
                                  evidence=ev_text[:8000], answer=str(answer)[:2000])
    try:
        raw = judge_fn(prompt)
    except Exception as ex:
        return SemanticAudit(None, [], [], 0.0, "audit_judge_error:%r" % (ex,))
    return _parse_audit(raw)


def _parse_audit(raw):
    if not raw:
        return SemanticAudit(None, [], [], 0.0, "empty_audit_response")
    s = raw if isinstance(raw, str) else str(raw)
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        try:
            d = json.loads(s[i:j + 1])
            at = d.get("addresses_task")
            hv = [h for h in (d.get("hard_violations") or []) if isinstance(h, dict)]
            rg = [g for g in (d.get("repairable_gaps") or []) if isinstance(g, dict)]
            conf = float(d.get("confidence", 0.0) or 0.0)
            return SemanticAudit(at if isinstance(at, bool) else None, hv, rg, conf)
        except Exception:
            pass
    return SemanticAudit(None, [], [], 0.0, "unparseable_audit_response")


# ---------------------------------------------------------------------------------------------------------
# CONSERVATIVE A/B SELECTOR (Layer-2). Compares the ORIGINAL answer A with a REVISED candidate B on PUBLIC
# dimensions only (no gold, no hidden reference). The adopt decision is made by the caller and is conservative
# by construction: B replaces A ONLY when the judge prefers B with a margin >= tau, the original critique is
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
                    "critique_resolved": bool(d.get("critique_resolved", False)),
                    "revised_new_hard_violation": bool(d.get("revised_new_hard_violation", False)),
                    "reason": d.get("reason")}
        except Exception:
            pass
    return default


def adopt_revised(comparison, tau=0.15):
    """CONSERVATIVE adopt rule: replace A with B ONLY if the judge prefers B by >= tau, the critique is
    resolved, and B adds no new hard violation. Anything else -> keep A. (tau is the user-set safety margin.)"""
    c = comparison or {}
    return bool(c.get("preferred") == "revised" and (c.get("margin") or 0.0) >= tau
                and c.get("critique_resolved") and not c.get("revised_new_hard_violation"))


# ---------------------------------------------------------------------------------------------------------
# GOAL-SPEC COMPILATION (P1.2). Restate the PUBLIC task goal in structured, checkable form -- never infer a
# hidden reference answer or the correct clinical decision; only what the task EXPLICITLY requires. Compiled
# once per task (frozen), so the harness can check "the action satisfied the task goal", not just "happened".
# ---------------------------------------------------------------------------------------------------------
_GOALSPEC_PROMPT = (
    "Restate, in STRUCTURED form, ONLY what the PUBLIC task below EXPLICITLY requires. Do NOT infer a hidden "
    "reference answer, the correct diagnosis, or the correct decision -- only restate the task's stated "
    "requirements. Reply STRICT JSON: {{\"requested_operation\": \"<what the agent must do>\", "
    '"required_effects": ["<observable effect the task requires>"], "required_fields": ["<element the output '
    'must contain>"], "forbidden_effects": ["<what must NOT happen>"], "success_observables": ["<what a '
    'correct completion makes observable>"]}}.\n\nGOAL:\n{goal}\n\nPUBLIC CONTEXT:\n{context}\n'
)


def compile_goal_spec(goal, public_context, judge_fn=None):
    """PUBLIC goal -> structured goal_spec dict (or None). Oracle-blind: reads only goal/context, never gold."""
    if not judge_fn or not goal:
        return None
    try:
        raw = judge_fn(_GOALSPEC_PROMPT.format(goal=str(goal)[:2000], context=str(public_context or "")[:2000]))
    except Exception:
        return None
    s = raw if isinstance(raw, str) else str(raw or "")
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        d = json.loads(s[i:j + 1])
    except Exception:
        return None
    def _l(k):
        v = d.get(k)
        return [str(x) for x in v if str(x).strip()] if isinstance(v, list) else ([str(v)] if v else [])
    return {"requested_operation": str(d.get("requested_operation") or "")[:300],
            "required_effects": _l("required_effects"), "required_fields": _l("required_fields"),
            "forbidden_effects": _l("forbidden_effects"), "success_observables": _l("success_observables")}


_ALIGN_PROMPT = (
    "A clinical agent is about to COMMIT a write. Using ONLY the public goal-spec, the current draft state, "
    "and the proposed action, decide whether the proposed commit SATISFIES the task's stated requirements. "
    "Do NOT infer the correct clinical decision -- only check whether the requested_operation, required "
    "fields, and required effects are present/addressed in the draft+action. Reply STRICT JSON: "
    '{{"aligned": true|false, "missing": ["<required field/effect not yet satisfied>"], "critique": "<short>"}}.'
    "\n\nGOAL-SPEC:\n{goal_spec}\n\nCURRENT DRAFT STATE:\n{state}\n\nPROPOSED ACTION:\n{action}\n"
)


def verify_goal_alignment(goal_spec, current_state, proposed_action, judge_fn=None):
    """Does the PROPOSED commit satisfy the public goal_spec (operation/fields/effects)? Oracle-blind; no
    judge/goal_spec -> aligned None (no opinion). Returns {aligned: True|False|None, missing: [...], critique}."""
    if not judge_fn or not goal_spec:
        return {"aligned": None, "missing": [], "critique": "no_check"}
    try:
        raw = judge_fn(_ALIGN_PROMPT.format(
            goal_spec=json.dumps(goal_spec, ensure_ascii=False)[:1500],
            state=json.dumps(current_state, default=str, ensure_ascii=False)[:3000],
            action=json.dumps(proposed_action, default=str, ensure_ascii=False)[:1000]))
    except Exception as ex:
        return {"aligned": None, "missing": [], "critique": "align_error:%r" % (ex,)}
    s = raw if isinstance(raw, str) else str(raw or "")
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        try:
            d = json.loads(s[i:j + 1])
            al = d.get("aligned")
            mi = [str(x) for x in (d.get("missing") or []) if str(x).strip()] if isinstance(d.get("missing"), list) else []
            return {"aligned": al if isinstance(al, bool) else None, "missing": mi, "critique": d.get("critique")}
        except Exception:
            pass
    return {"aligned": None, "missing": [], "critique": "unparseable"}


# ---------------------------------------------------------------------------------------------------------
# SCOPED REPAIR (replaces the NL obligation list). The judge inspects the proposed output against the public
# goal-spec + the current draft state and returns ONLY localized defects, each naming an exact target_path
# (drawn from the state keys it is shown), the smallest repair operation, and the content to PRESERVE. A
# finding with no concrete target/change is dropped by parse_findings -> the harness never emits a vague
# "write a triage note" REVISE. Oracle-blind: never infers the correct clinical decision.
# ---------------------------------------------------------------------------------------------------------
_SCOPED_REPAIR_PROMPT = (
    "Inspect the PROPOSED output against the public goal specification and the current draft state. Return "
    "ONLY concrete, localized defects. Do NOT restate broad task obligations. A finding is valid ONLY if you "
    "can name: (1) the exact target location that exists in the draft/state; (2) the exact missing, "
    "unsupported, or conflicting content; (3) the smallest permitted repair; (4) content that must be "
    "preserved. Do NOT infer the correct clinical decision. Reply STRICT JSON: "
    '{{"aligned": true|false, "findings": [{{"target_type": "field|resource_path|claim|action", '
    '"target_path": "<key that appears in the DRAFT STATE/GOAL-SPEC>", "defect_type": '
    '"missing|insufficient_content|unsupported|conflicting|wrong_operation", "repair_operation": '
    '"ADD|EDIT|REMOVE|REPLACE|VERIFY", "required_change": "<the concrete content to add/fix>", '
    '"protected_paths": ["<paths whose existing content must NOT change>"], "preserve_requirements": '
    '["<substantive content to retain>"], "evidence_refs": ["<evidence supporting the change>"], '
    '"confidence": 0.0}}]}}. Do NOT emit a finding such as "write a triage note" or "document reasoning" -- '
    "state WHAT concrete content is absent and WHERE. Use target_path values that match the keys shown.\n\n"
    "GOAL-SPEC:\n{goal_spec}\n\nCURRENT DRAFT STATE:\n{state}\n\nPROPOSED OUTPUT:\n{candidate}\n"
)


def scoped_goal_findings(goal_spec, state, candidate, judge_fn=None, task_id="t", rule_id="scoped_repair", surface=None):
    """PUBLIC goal_spec + draft state + proposed output -> [RepairFinding] (localized, structured). No judge /
    no goal_spec / unparseable -> [] (stay silent). Oracle-blind."""
    if not judge_fn or not goal_spec:
        return []
    from ..repair_surface import path_space
    _root = surface.root(state, candidate) if surface is not None else state
    _paths = path_space(_root)
    _state_str = (json.dumps(_root, default=str, ensure_ascii=False)[:2600]
                  + "\n\nADDRESSABLE PATHS (target_path MUST be EXACTLY one of these existing paths, or a "
                    "new child 'parent.newkey' of one; do NOT invent a path that is not listed here): "
                  + json.dumps(_paths, ensure_ascii=False)[:1200])
    try:
        raw = judge_fn(_SCOPED_REPAIR_PROMPT.format(
            goal_spec=json.dumps(goal_spec, ensure_ascii=False)[:1500],
            state=_state_str,
            candidate=json.dumps(candidate, default=str, ensure_ascii=False)[:1500]))
    except Exception:
        return []
    s2 = raw if isinstance(raw, str) else str(raw or "")
    i, j = s2.find("{"), s2.rfind("}")
    if i < 0 or j <= i:
        return []
    try:
        d = json.loads(s2[i:j + 1])
    except Exception:
        return []
    from ..repair import parse_findings
    return parse_findings(d, task_id, rule_id)


# ---------------------------------------------------------------------------------------------------------
# EVIDENCE COVERAGE judge steps. (1) decompose the final answer into atomic, TYPED claims with a concrete
# target; (2) for margin claims, batch-decide whether the agent OWN observations support them. Oracle-blind:
# classify/locate only, never judge clinical correctness, never name a tool (the affordance registry does).
# ---------------------------------------------------------------------------------------------------------
_CLAIM_DECOMP_PROMPT = (
    "Decompose the agent FINAL ANSWER into atomic claims. Classify EACH and, for perceptual claims, name "
    "the concrete target. Reply STRICT JSON: "
    '{{"claims": [{{"text": "...", "claim_type": "perceptual|interpretive|background|recommendation", '
    '"region": "<target or null>", "modality": "<modality or null>", "attribute": "<feature or null>"}}]}}. '
    "Definitions: perceptual = a directly-observed image feature at a location; interpretive = a judgment "
    "derived from features; background = general medical knowledge; recommendation = a suggested action. Do "
    "NOT judge correctness; only classify and locate.\n\nFINAL ANSWER:\n{answer}\n\nPUBLIC CONTEXT:\n{context}\n"
)


def decompose_claims(answer, public_context, judge_fn=None, task_id="t"):
    """Final answer -> [observation.Claim]. No judge / unparseable -> []."""
    if not judge_fn or not answer:
        return []
    try:
        raw = judge_fn(_CLAIM_DECOMP_PROMPT.format(answer=str(answer)[:2500], context=str(public_context or "")[:1500]))
    except Exception:
        return []
    t = raw if isinstance(raw, str) else str(raw or "")
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return []
    try:
        d = json.loads(t[i:j + 1])
    except Exception:
        return []
    from ..observation import Claim, CLAIM_TYPES
    out = []
    for k, c in enumerate(d.get("claims") or []):
        if not isinstance(c, dict):
            continue
        ct = str(c.get("claim_type") or "").strip().lower()
        if ct not in CLAIM_TYPES:
            ct = "interpretive"
        def _v(key):
            x = c.get(key)
            return str(x).strip() if x not in (None, "", "null") else None
        out.append(Claim(claim_id="claim-%d" % k, idx=k, claim_type=ct, text=str(c.get("text") or "")[:300],
                         region=_v("region"), modality=_v("modality"), attribute=_v("attribute")))
    return out


_CLAIM_SUPPORT_PROMPT = (
    "For EACH claim, decide whether the agent OWN listed observations SUPPORT it. Use ONLY the observations "
    "shown; do not use outside knowledge and do not judge clinical correctness. Reply STRICT JSON: "
    '{{"support": {{"<claim_id>": true|false}}}}.'
    "\n\nOBSERVATIONS:\n{obs}\n\nCLAIMS:\n{claims}\n"
)


def claim_semantic_support(claims, observation_summaries, judge_fn=None):
    """{claim_id: bool} for the given margin claims. No judge -> {} (caller stays silent -> conservative)."""
    if not judge_fn or not claims:
        return {}
    cl = "\n".join("%s: %s" % (c.claim_id, c.text or ("%s @ %s" % (c.attribute, c.region))) for c in claims)
    try:
        raw = judge_fn(_CLAIM_SUPPORT_PROMPT.format(obs=str(observation_summaries)[:2500], claims=cl[:2000]))
    except Exception:
        return {}
    t = raw if isinstance(raw, str) else str(raw or "")
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return {}
    try:
        d = json.loads(t[i:j + 1])
        sup = d.get("support") or {}
        def _b(v):  # strict: judge may return the STRING "false" -> bool("false") is True (wrong)
            return v is True or (isinstance(v, str) and v.strip().lower() in ("true", "yes", "1"))
        return {str(k): _b(v) for k, v in sup.items()}
    except Exception:
        return {}
