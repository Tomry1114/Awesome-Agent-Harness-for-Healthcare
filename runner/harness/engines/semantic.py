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


def audit_answer(task_goal, public_context, answer, evidence, judge_fn=None):
    """Adequacy audit -> SemanticAudit{addresses_task, hard_violations[], repairable_gaps[], confidence}.
    No judge -> empty audit (fail-safe: no fabricated defects). Evidence is the SAME selector-filtered,
    VALIDATED set the claim-support check sees."""
    if not judge_fn:
        return SemanticAudit(None, [], [], 0.0, "audit_judge_unavailable")
    ev_text = _format_evidence(evidence)
    prompt = _AUDIT_PROMPT.format(task_goal=str(task_goal or "(not provided)")[:2000],
                                  public_context=str(public_context or "(none)")[:3000],
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
