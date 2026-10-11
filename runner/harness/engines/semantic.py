"""Semantic engine — judge-backed checks the deterministic engine cannot do (claim<->evidence support).

The judge is INJECTED as a `judge_fn(prompt:str) -> str|None` (the kernel builds it from the gateway with
an INDEPENDENT judge model, budgeted by max_semantic_checks). This keeps capabilities model-agnostic and
unit-testable (tests inject a fake judge). Fail-SAFE: no judge -> UNKNOWN (never fabricate a pass), and a
low-confidence verdict -> caller ESCALATEs rather than silently allowing.
"""
import hashlib
import json

UNKNOWN = "UNKNOWN"

# Relation tri-state: distinguish a real CONTRADICTION (answer conflicts with the evidence) from
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
            # Reconcile relation <-> supported. A {"supported": bool} judge reply carries no
            # relation -> derive it (true->supported, false->contradicted). A {"relation": ...} judge reply
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
    # DERIVE the discriminator from the GOAL, not just re-describe the current answer's region. When
    # the goal has multiple required elements (e.g. "identify the tissue AND determine the stain"), force a
    # per-element coverage check so an element the answer never addresses becomes a TARGETED 'unanswered' gap
    # -> the next acquisition pursues THAT element, instead of re-observing what the answer already covers.
    _gs = ""
    if goal_spec:
        _req = [str(x) for x in ((goal_spec.get("required_fields") or [])
                                 + (goal_spec.get("success_observables") or [])
                                 + (goal_spec.get("required_effects") or [])) if str(x).strip()]
        _enum = "; ".join(dict.fromkeys(_req))   # dedup, keep order
        _gs = ("\n\nThe task EXPLICITLY requires (structured): " + json.dumps(goal_spec, ensure_ascii=False)
               + (("\n\nMANDATORY PER-ELEMENT COVERAGE: the answer must address EACH required element: ["
                   + _enum + "]. For EVERY element the FINAL ANSWER does not explicitly answer, emit a REPAIRABLE "
                   "gap of type 'unanswered' whose critique NAMES that exact element (so the agent knows precisely "
                   "what to go determine). Do this even if the OTHER elements are answered well.") if _enum else
                  "\nAlso flag, as a REPAIRABLE gap, any required_effect/required_field/requested_operation the "
                  "answer does not satisfy."))
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
# GOAL-SPEC COMPILATION. Restate the PUBLIC task goal in structured, checkable form -- never infer a
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



# ---------------------------------------------------------------------------------------------------------
# SCOPED REPAIR. The judge inspects the proposed output against the public goal-spec + the current draft
# state and returns ONLY localized defects, each naming an exact target_path (drawn from the state keys it is
# shown), the smallest repair operation, and the content to PRESERVE. A finding with no concrete
# target/change is dropped by parse_findings -> the harness never emits a vague "write a triage note" REVISE.
# Oracle-blind: never infers the correct clinical decision.
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


# ---- semantic judge result cache (content-addressed) ----------------------------------------------
# The same text can be decomposed many times within one task: two independent modules
# (evidence_coverage / recovery.medcta) each decompose the same answer, and repeated checks re-send identical
# prompts. A cache hit returns a byte-identical result, so downstream decisions (region coverage, support,
# admission) are unchanged -- the cache only removes duplicate judge calls, it never changes a criterion.
#
# Scope: the cache belongs to one task. run.py calls sem_cache_reset(task_id) at the start of _run_task, which
# clears the cache and records the task; key = function name + that task_id + judge model (judge_fn.judge_model,
# which every HarnessContext judge carries) + the full prompt sent to the judge. This keeps a later task in the
# same process (replay, run_batch) from reading results cached for a previous task or for a different judge.
_SEM_CACHE = {}
_SEM_CACHE_ORDER = []
_SEM_CACHE_MAX = 256
_SEM_SCOPE = {"task_id": None}          # the task the cache belongs to (sem_cache_reset)
SEM_CACHE_STATS = {"hit": 0, "miss": 0}


def _sem_cache_key(fn_name, judge_fn, prompt):
    h = hashlib.sha256()
    for x in (fn_name, _SEM_SCOPE["task_id"], getattr(judge_fn, "judge_model", None), prompt):
        h.update(str(x).encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()


def _sem_cache_get(key):
    if key in _SEM_CACHE:
        SEM_CACHE_STATS["hit"] += 1
        return _SEM_CACHE[key], True
    SEM_CACHE_STATS["miss"] += 1
    return None, False


def _sem_cache_put(key, val):
    if key not in _SEM_CACHE:
        _SEM_CACHE_ORDER.append(key)
        if len(_SEM_CACHE_ORDER) > _SEM_CACHE_MAX:
            _SEM_CACHE.pop(_SEM_CACHE_ORDER.pop(0), None)
    _SEM_CACHE[key] = val
    return val


def sem_cache_reset(task_id=None):
    """Clear the cache and record the new task on a task switch -- the cache is partitioned per task and never
    reused across tasks (called at the start of run.py _run_task)."""
    _SEM_CACHE.clear()
    del _SEM_CACHE_ORDER[:]
    _SEM_SCOPE["task_id"] = task_id


def decompose_claims(answer, public_context, judge_fn=None, task_id="t"):
    """Final answer -> [observation.Claim]. No judge / unparseable -> []. `task_id` is kept for callers; the cache is
    scoped by sem_cache_reset."""
    if not judge_fn or not answer:
        return []
    _prompt = _CLAIM_DECOMP_PROMPT.format(answer=str(answer)[:2500], context=str(public_context or "")[:1500])
    _ck = _sem_cache_key("decompose_claims", judge_fn, _prompt)
    _hit, _ok = _sem_cache_get(_ck)
    if _ok:
        return list(_hit)                     # same text -> reuse the first result, no new request
    try:
        raw = judge_fn(_prompt)
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
    _sem_cache_put(_ck, list(out))
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
    _prompt = _CLAIM_SUPPORT_PROMPT.format(obs=str(observation_summaries)[:2500], claims=cl[:2000])
    _ck = _sem_cache_key("claim_semantic_support", judge_fn, _prompt)
    _hit, _ok = _sem_cache_get(_ck)
    if _ok:
        return dict(_hit)                     # same claims + same evidence -> reuse, no new request
    try:
        raw = judge_fn(_prompt)
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
        return _sem_cache_put(_ck, {str(k): _b(v) for k, v in sup.items()})
    except Exception:
        return {}


_DISCRIMINATOR_PROMPT = (
    "An agent gave the answer below to a perceptual/diagnostic task. Decide whether a SINGLE additional targeted "
    "observation could still change the answer. WITHOUT outside knowledge or a gold answer, name the TWO most "
    "plausible interpretations and the SINGLE most discriminating OBSERVABLE feature (region + attribute) a "
    "perception tool could check to tell them apart. If that feature was ALREADY observed in OBSERVATIONS ALREADY "
    "MADE below, OR the answer is already certain / non-perceptual, return null region (do NOT invent a target "
    "that adds nothing). Reply STRICT JSON: "
    '{{"hypotheses": ["<h1>", "<h2>"], "region": "<region or null>", "attribute": "<feature or null>"}}.'
    "{required}"
    "\n\nGOAL:\n{goal}\n\nOBSERVATIONS ALREADY MADE:\n{obs}\n\nANSWER:\n{answer}\n"
)


_EVCLASS_PROMPT = (
    "An agent is answering a clinical/perceptual question and ONE required element is still UNRESOLVED. Decide "
    "which KIND of evidence would resolve it:\n"
    "- 'visual': needs LOOKING harder at the image (a finding/region/measurement present in the pixels).\n"
    "- 'knowledge': needs EXTERNAL reference knowledge NOT determinable by looking harder -- what a "
    "stain/marker/sign LOOKS like, a definition, epidemiology/rarity (common vs rare), a classification, the "
    "clinical meaning of an observed feature.\n"
    "- 'text': needs reading printed TEXT/labels in the image (panel label, gene name, specimen/stain name, "
    "axis or value annotation, group/stage/timepoint label).\n"
    "- 'numeric': needs a CALCULATION over values ALREADY observed (ratio, percentage, difference, unit "
    "conversion, a simple formula result) -- NOT re-looking and NOT searching.\n"
    "Reply STRICT JSON: {{\"need\": \"visual|knowledge|text|numeric\", \"query\": \"<if knowledge: a concise "
    "web-search query GROUNDED IN THE OBSERVED FEATURES AND ANY OCR-READ LABELS / GENE NAMES / ANNOTATIONS "
    "(which often carry the decisive context) that would return the needed fact; else empty>\", "
    "\"expression\": \"<if numeric: an arithmetic expression over the observed values, e.g. "
    "(value_a - value_b) / value_b * 100; else empty>\"}}.\n\n"
    "UNRESOLVED ELEMENT:\n{element}\n\nOBSERVED SO FAR:\n{obs}\n\nCURRENT ANSWER:\n{answer}\n")

_GVERIFY_PROMPT = (
    "A medical visual-QA task. Given the PUBLIC QUESTION, the OBSERVED FEATURES (including any OCR-read labels/"
    "gene names), and the agent CURRENT ANSWER, decide whether the answer makes a knowledge-checkable "
    "IDENTIFICATION that EXTERNAL reference knowledge can verify -- ONLY: tissue type, organ, canonical "
    "structure identity, stain, or diagnosis. Produce ONE web-search query to INDEPENDENTLY establish what the "
    "observed features are, so the agent can cross-check its claim. THE QUERY MUST: be grounded in the PUBLIC "
    "QUESTION target + the OBSERVED FEATURES + OCR labels/gene names; and MUST NOT contain the agent CURRENT "
    "candidate identification NAME (including it biases the search toward CONFIRMING a possibly-wrong answer). "
    "Reply an EMPTY query when external knowledge CANNOT settle it from this specific image -- i.e. imaging "
    "plane/view/orientation (geometry), presence/absence of a finding, which panel is stronger, a measurement/"
    "reading, or an uncertain/negative answer -- those need the pixels, not a search. Reply STRICT JSON: "
    "{{\"query\": \"<independent verification query, or empty>\"}}.\n\nPUBLIC QUESTION:\n{goal}\n\n"
    "OBSERVED FEATURES:\n{obs}\n\nAGENT CURRENT ANSWER:\n{answer}\n")

_PINCON_PROMPT = (
    "An agent is analysing a medical image and recorded these OBSERVATIONS. Detect an INTERNAL INCONSISTENCY: "
    "an observation whose DESCRIBED visual features do NOT match the STRUCTURE/identity it names or concludes "
    "(e.g. describes visual features characteristic of one structure/tissue type yet NAMES a different one; "
    "or reports a finding as present in one observation and absent in another). If such an "
    "inconsistency exists, produce a web-search query GROUNDED IN THE DESCRIBED FEATURES (NOT the agent label) "
    "that would reveal what those features actually are. If the observations are mutually consistent, reply "
    "inconsistent=false. Reply STRICT JSON: {{\"inconsistent\": true|false, \"query\": \"<grounded feature "
    "query, or empty>\"}}.\n\nOBSERVATIONS:\n{obs}\n")

_ENTITY_PROMPT = (
    "An agent analysing a scientific / medical image recorded these OBSERVATIONS -- including OCR-read text "
    "labels and any web searches it already ran. Some labels denote SCIENTIFIC ENTITIES -- a gene, protein, "
    "biomarker, antibody, cell line, genotype, or specimen identifier -- whose canonical identity carries "
    "DECISIVE context (the entity often reveals which organ / tissue / disease model the figure is about). "
    "Decide whether any such entity label CANONICAL IDENTITY and biological context remain UNRESOLVED by the "
    "observations/searches (not searched, or searched only GENERICALLY as image text). If so, produce ONE "
    "focused web-search query by treating the label AS A SCIENTIFIC ENTITY: the label followed by generic "
    "descriptors such as gene protein biomarker knockout genotype phenotype organ tissue histology, asking for "
    "its canonical identity AND biological/tissue context. Do NOT assume any specific identity. If every entity "
    "label is already resolved, or there is none, reply empty. Reply STRICT JSON: {{\"entity\": \"<label or "
    "empty>\", \"query\": \"<entity-focused query, or empty>\"}}.\n\nOBSERVATIONS:\n{obs}\n")


_SYNTH_PROMPT = (
    "An agent gave a FINAL answer to the PUBLIC QUESTION. You are given the EVIDENCE it already gathered "
    "(observations + search results). Decide whether the answer core conclusion is GROUNDED in that evidence, "
    "or whether it IGNORES or CONTRADICTS decisive evidence that points to a DIFFERENT conclusion. Reply "
    "grounded=FALSE ONLY IF the evidence the agent itself gathered clearly points AWAY from its conclusion -- "
    "e.g. a search result identifying an entity or context that implies a different answer, or an observed "
    "feature its conclusion directly contradicts. If the conclusion is merely UNPROVEN but NOT contradicted, "
    "or the evidence is simply CONSISTENT with it, reply grounded=TRUE (do not nag). Judge ONLY the agent OWN "
    "gathered evidence vs its conclusion, never any external truth. When grounded=false, QUOTE the specific "
    "unreconciled evidence WITHOUT stating what the answer should be. Reply STRICT JSON: "
    "{{\"grounded\": true|false, \"unreconciled\": \"<the specific gathered evidence the conclusion fails to "
    "reconcile, quoted; or empty>\"}}.\n\nPUBLIC QUESTION:\n{goal}\n\nANSWER:\n{answer}\n\n"
    "GATHERED EVIDENCE:\n{ev}\n")

def synthesis_grounding_check(answer, evidence, goal, judge_fn=None):
    """Synthesis grounding check: is the answer GROUNDED in the evidence the agent ALREADY gathered, or did it
    IGNORE decisive evidence (e.g. its own search result) / reach a conclusion the evidence contradicts? Returns
    {grounded, unreconciled}. Oracle-blind (agent's own evidence vs its conclusion, never gold). Names the
    unreconciled evidence so the agent can RE-SYNTHESISE -- the harness never states the answer. Fail-OPEN: no
    judge / parse-fail -> grounded=True (do not nag on uncertainty)."""
    if not judge_fn or not str(answer or "").strip() or not evidence:
        return {"grounded": True, "unreconciled": ""}
    ev = "\n".join("- %s" % o for o in list(evidence)[:14]) or "(none)"
    try:
        d = _json_obj(judge_fn(_SYNTH_PROMPT.format(goal=str(goal or "")[:800], answer=str(answer)[:1000], ev=ev[:3000])))
    except Exception:
        return {"grounded": True, "unreconciled": ""}
    if not isinstance(d, dict):
        return {"grounded": True, "unreconciled": ""}
    return {"grounded": _as_bool(d.get("grounded"), True), "unreconciled": str(d.get("unreconciled") or "").strip()[:500]}


def detect_unexploited_entity_label(observations, judge_fn=None):
    """Search-QUALITY intervention: did the agent OCR an ENTITY label (gene/marker/specimen) whose decisive
    context it has NOT established -- because it did not search it, or searched it only GENERICALLY as image
    text (label searched only as image text, not as the entity)? Returns {entity, query} with an
    ENTITY-focused query that surfaces the entity canonical identity + tissue context. Fail-safe: no judge /
    parse-fail -> empty."""
    if not judge_fn or not observations:
        return {"entity": "", "query": ""}
    obs = "\n".join("- %s" % o for o in list(observations)[:14]) or "(none)"
    try:
        d = _json_obj(judge_fn(_ENTITY_PROMPT.format(obs=obs[:2400])))
    except Exception:
        return {"entity": "", "query": ""}
    if not isinstance(d, dict):
        return {"entity": "", "query": ""}
    return {"entity": str(d.get("entity") or "").strip()[:40], "query": str(d.get("query") or "").strip()[:300]}



def grounded_verification_query(answer, goal, observations=None, judge_fn=None):
    """Adapter (MedCTA): for a CONFIDENT identification answer, build a web-search query GROUNDED in the
    observed features to cross-check the identification against external knowledge. Returns '' when the answer
    is not a verifiable identification. Oracle-blind (features + answer, never gold)."""
    if not judge_fn or not str(answer or "").strip():
        return ""
    obs = "\n".join("- %s" % o for o in (observations or [])[:10]) or "(none)"
    try:
        d = _json_obj(judge_fn(_GVERIFY_PROMPT.format(goal=str(goal or "")[:600], obs=obs[:1400], answer=str(answer)[:1000])))
    except Exception:
        return ""
    return (str(d.get("query") or "").strip()[:300]) if isinstance(d, dict) else ""


def classify_evidence_need(element, answer, observations=None, judge_fn=None):
    """Oracle-blind (element + own observations, never gold): what evidence CLASS resolves the unresolved
    element -- 'visual' | 'knowledge' | 'text' -- and, for knowledge, a web-search query grounded in the
    observed features. Fail-safe: no judge / parse-fail -> ('visual', '') so perception stays the default."""
    if not judge_fn or not element:
        return ("visual", "", "")
    obs = "\n".join("- %s" % o for o in (observations or [])[:10]) or "(none)"
    try:
        d = _json_obj(judge_fn(_EVCLASS_PROMPT.format(element=str(element)[:400], obs=obs[:1400],
                                                      answer=str(answer or "")[:800])))
    except Exception:
        return ("visual", "", "")
    if not isinstance(d, dict):
        return ("visual", "", "")
    need = str(d.get("need") or "visual").strip().lower()
    if need not in ("visual", "knowledge", "text", "numeric"):
        need = "visual"
    return (need, str(d.get("query") or "").strip()[:300], str(d.get("expression") or "").strip()[:200])


def elicit_discriminator(answer, goal, observations=None, judge_fn=None, required_element=None):
    """answer -> {hypotheses[2], region, attribute} (oracle-blind: reads only the agent answer/goal). region
    None when the answer is certain / non-perceptual. When the adequacy audit already named an
    UNANSWERED required element (e.g. "determine the stain"), TARGET the observable needed to determine THAT
    element -- do not re-derive a discriminator from the answer's current (wrong-surface) content."""
    if not judge_fn or not answer:
        return None
    try:
        obs = "\n".join("- %s" % o for o in (observations or [])[:12]) or "(none)"
        _req = ("\n\nPRIORITY -- the answer specifically FAILS TO DETERMINE this REQUIRED element: '"
                + str(required_element)[:300] + "'. Name the region + attribute a perception tool must check to "
                "DETERMINE THAT ELEMENT directly (target it), not a generic re-look of what is already answered."
                ) if required_element else ""
        raw = judge_fn(_DISCRIMINATOR_PROMPT.format(goal=str(goal or "")[:1200], obs=obs[:1500],
                                                    answer=str(answer)[:1200], required=_req))
    except Exception:
        return None
    t = raw if isinstance(raw, str) else str(raw or ""); i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        d = json.loads(t[i:j + 1])
    except Exception:
        return None
    def _v(k):
        x = d.get(k)
        return str(x).strip() if x not in (None, "", "null") and str(x).strip().lower() != "null" else None
    return {"hypotheses": [str(x) for x in (d.get("hypotheses") or [])][:2], "region": _v("region"),
            "attribute": _v("attribute")}


def _as_bool(v, default=False):
    """Strict JSON boolean: a judge that returns the STRING "false" / "no" / "0" must be
    False, not a truthy non-empty string. bool("false") is True in Python, which would corrupt every
    supported / critique_resolved / preserved / changed / uncertain flag parsed straight from judge JSON."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1", "y", "t")
    return default


def _json_obj(raw):
    s = raw if isinstance(raw, str) else str(raw or "")
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(s[i:j + 1])
        except Exception:
            return None
    return None


# =====================================================================================================
# INTENT-SCOPED CONTEXT. A deliverable does NOT require every policy obligation -- only those tied to the
# clinical ACTIONS the deliverable actually expresses. Oracle-blind: reads the PUBLIC goal + the agent's own
# proposed deliverable content + the candidate obligations' resources; never gold. A plan that starts/changes a
# medication makes AllergyIntolerance/MedicationRequest relevant; a pure imaging-follow-up plan does not.
# =====================================================================================================

_RELEVANCE_PROMPT = (
    "A clinical DELIVERABLE (a plan/note) is about to be committed. Decide which of the CANDIDATE required-"
    "context records are RELEVANT to the clinical ACTIONS the deliverable actually expresses (e.g. a record is "
    "relevant only if the plan makes a decision that depends on it). Do NOT judge correctness; do NOT require a "
    "record the plan's actions do not depend on. Reply STRICT JSON: {{\"relevant\": [\"<obligation_id>\"...]}}.\n\n"
    "PUBLIC GOAL:\n{goal}\n\nDELIVERABLE CONTENT:\n{content}\n\nCANDIDATE REQUIRED-CONTEXT (id -> resource):\n{obs}\n")


def select_relevant_obligations(goal, content, obligations, judge_fn=None):
    """obligations = [(oid, resource)]. Return the subset whose record the deliverable's clinical actions
    actually depend on. No judge / parse-fail / no content -> return ALL (fail-safe: do not silently drop a
    required check). Empty obligations -> []."""
    obs = [(oid, res) for (oid, res) in (obligations or []) if oid]
    if not obs:
        return []
    if not judge_fn or not content:
        return obs
    listing = "\n".join("%s -> %s" % (oid, res) for (oid, res) in obs)
    prompt = _RELEVANCE_PROMPT.format(goal=str(goal or "")[:1200], content=str(content or "")[:4000], obs=listing)
    try:
        d = _json_obj(judge_fn(prompt))
    except Exception:
        return obs
    if not isinstance(d, dict) or "relevant" not in d:
        return obs
    keep = set(str(x) for x in (d.get("relevant") or []))
    sel = [(oid, res) for (oid, res) in obs if oid in keep]
    return sel       # may be empty -> the deliverable's actions depend on no required record


# =====================================================================================================
# PLAN COMPLETENESS (the COMPLETE recovery for a goal-required-but-missing commitment slot).
# A deliverable can FAIL not because a claim is wrong but because a slot the PUBLIC GOAL requires is simply
# ABSENT (no follow-up section, no workup, no monitoring plan). This is the F->T path least dependent on a
# chance-positive record: it needs only the goal's own structure + the draft. Oracle-blind (reads ONLY the
# public goal + the agent's draft; never gold/checkpoint). Localized: fills ONLY the missing slot, never
# rewrites or edits existing content. All three functions FAIL-SAFE toward "do nothing" so the module can
# never fabricate a requirement, drop a present slot, or over-fire.
# =====================================================================================================

_REQ_SLOTS_PROMPT = (
    "You are decomposing a clinical task's DELIVERABLE structure. From the GOAL below, list ONLY the sections a "
    "COMPLETE deliverable is EXPLICITLY required to contain -- i.e. components the goal itself asks for (or that "
    "a structure the goal names by title requires). Do NOT invent optional extras, and do NOT list a section "
    "the goal does not clearly require. If the goal is generic and mandates no specific sections, return an empty "
    "list. Do NOT write any section's content. Reply STRICT JSON: "
    '{{"required_slots": [{{"slot_id": "<lower_snake_case>", "description": "<short: what this section must cover>"}}]}}.'
    "\n\nGOAL:\n{goal}\n")


def extract_required_slots(goal, judge_fn=None):
    """Required commitment slots implied by the PUBLIC goal. [{'slot_id','description'}]. Fail-safe: [] on
    no-judge / parse-fail / empty (=> no completeness gap => never fabricates a requirement)."""
    if not judge_fn or not str(goal or "").strip():
        return []
    try:
        d = _json_obj(judge_fn(_REQ_SLOTS_PROMPT.format(goal=str(goal)[:2500])))
    except Exception:
        return []
    if not isinstance(d, dict):
        return []
    out = []
    for s in (d.get("required_slots") or []):
        if not isinstance(s, dict):
            continue
        sid = str(s.get("slot_id") or "").strip().lower().replace(" ", "_")
        if sid:
            out.append({"slot_id": sid, "description": str(s.get("description") or sid)[:300]})
    # de-dup by slot_id, order preserved
    seen, uniq = set(), []
    for s in out:
        if s["slot_id"] not in seen:
            seen.add(s["slot_id"]); uniq.append(s)
    return uniq


_PRESENT_SLOTS_PROMPT = (
    "A REQUIRED-section checklist and an agent's DRAFT deliverable are below. For EACH required section decide "
    "whether the draft ADDRESSES it with substantive content (present) or leaves it ABSENT/empty (absent). Judge "
    "only presence of substantive content, NOT correctness. Reply STRICT JSON: "
    '{{"present": ["<slot_id>"...], "absent": ["<slot_id>"...]}}.'
    "\n\nREQUIRED SECTIONS:\n{listing}\n\nDRAFT:\n{content}\n")


def parse_present_slots(content, required_slots, judge_fn=None):
    """Which required slots the draft already ADDRESSES. Returns [dict slot] present (slot_id + required_by_goal).
    Fail-safe: treat ALL required as PRESENT on no-judge / no-content / parse-fail (=> missing set empty =>
    never over-fires)."""
    req = [s for s in (required_slots or []) if s.get("slot_id")]
    if not req:
        return []
    all_present = [{"slot_id": s["slot_id"], "value": "<present>", "required_by_goal": True} for s in req]
    if not judge_fn or not str(content or "").strip():
        return all_present
    listing = "\n".join("%s -> %s" % (s["slot_id"], s.get("description", "")) for s in req)
    try:
        d = _json_obj(judge_fn(_PRESENT_SLOTS_PROMPT.format(listing=listing, content=str(content)[:6000])))
    except Exception:
        return all_present
    if not isinstance(d, dict) or "present" not in d:
        return all_present
    pset = set(str(x).strip().lower().replace(" ", "_") for x in (d.get("present") or []))
    return [{"slot_id": s["slot_id"], "value": "<present>", "required_by_goal": True}
            for s in req if s["slot_id"] in pset]


_SLOT_GEN_PROMPT = (
    "You are a clinical documentation specialist COMPLETING a deliverable. The draft is MISSING one required "
    "section. Produce ONLY that section, consistent with the GOAL and the draft's EXISTING plan. Ground it in "
    "facts already established in the draft; do NOT invent patient specifics (labs, doses, diagnoses) not present "
    "in the draft, and do NOT restate or modify any other section. If you cannot produce a defensible section "
    "from the available material, return an empty content string. Reply STRICT JSON: "
    '{{"section_title": "<short heading>", "content": "<the section body only>"}}.'
    "\n\nGOAL:\n{goal}\n\nMISSING SECTION:\n{slot_id} -- {desc}\n\nEXISTING DRAFT:\n{content}\n")


def generate_slot_content(goal, draft, slot, judge_fn=None):
    """Structured specialist: content for ONE missing required slot, grounded in goal + draft. Returns
    {'section_title','content'} or None (no judge / parse-fail / empty body -- never a fabricated section)."""
    if not judge_fn:
        return None
    sid = str((slot or {}).get("slot_id") or "").strip()
    desc = str((slot or {}).get("description") or sid)
    if not sid:
        return None
    try:
        d = _json_obj(judge_fn(_SLOT_GEN_PROMPT.format(goal=str(goal or "")[:1500], slot_id=sid,
                                                       desc=desc[:300], content=str(draft or "")[:5000])))
    except Exception:
        return None
    if not isinstance(d, dict):
        return None
    body = str(d.get("content") or "").strip()
    if not body:
        return None
    title = str(d.get("section_title") or sid.replace("_", " ").title()).strip()
    return {"section_title": title, "content": body}
