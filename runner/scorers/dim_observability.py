#!/usr/bin/env python3
"""
Clinical Auditability dimension (experimental tier) — benchmark-AGNOSTIC.

Question: did the EXECUTION SYSTEM make the world observable to the deciding agent? i.e. was task
evidence and were failures actually DELIVERED, at fidelity, into the agent's context — and were they
visible / taken up downstream. Tool TYPE is not guessed from tool-name keywords ("search"/"read"/"ocr"...)
and uptake is not measured by lexical long-word overlap. Here we consume ONLY the substrate EvidenceView
fields the plugin already populated:
    delivered_to_agent, delivery_fidelity, error_visible, acknowledged (optional)
plus the SemanticTrace (only to find the downstream agent surface for an optional uptake fallback).
NO benchmark name / tool literal / image / DOM / FHIR string appears in the scoring logic.

THREE DISTINCT DELIVERY FACTS — a single number would hide the layers, and the pair
delivered_to_agent=False AND error_visible=True on the SAME unit would look contradictory. That pair is NOT
contradictory once split: the RESULT content did not reach the agent, but the FAILURE signal did. So we
derive, per evidence unit, three independent facts from the substrate fields
{delivered_to_agent, error_visible, delivery_fidelity}:
  observation_rendered      = ANYTHING at all was shown to the agent for this step
                              (the result reached it, OR an error/failure message reached it).
                              False ONLY when the renderer dropped everything (delivered=False AND
                              error_visible=False).
  result_evidence_delivered = the actual tool RESULT content reached the agent  (== delivered_to_agent).
  error_signal_delivered    = the failure signal reached the agent              (error_visible on an
                              error-bearing unit).

LAYERS — reported SEPARATELY (each its own number) AND folded into a composite (applicable-only — a
layer with no opportunity is status=not_applicable and EXCLUDED from the mean, so a clean run never
collects a vacuous 1.0):
  delivery           = result_evidence_delivered / total units   (NOT 'no error'; the actual content
                       reaching the agent)                        (always applies if units>0)
  fidelity           = mean delivery_fidelity over DELIVERED units (applies if any delivered)
  error_transparency = error_signal_delivered / error-bearing units  (N/A if no error-bearing units)
  uptake (PROXY)     = delivered units referenced/acknowledged downstream   (N/A if no downstream text
                       AND no unit carries an `acknowledged` flag) — tier=proxy, kept explicit.

The headline score is the operator-facing Diagnostic Clinical Auditability (see diagnostic_observability);
the layers above are reported alongside it as diagnostics.

Entry: observability(evidence, sem_trace) -> {score, status, coverage, reportable, layers, delivery_facts,
delivery / fidelity / error_transparency / uptake (each its own number), composite, submetrics...}
"""
import re

_SYS_W = {"delivery": 0.45, "fidelity": 0.35, "error_transparency": 0.20}
_TIER = "experimental"
# uptake is a weaker, lexical/heuristic signal — flagged proxy so the composite consumer can see it.
_LAYER_TIER = {"delivery": "experimental", "fidelity": "experimental",
               "error_transparency": "experimental", "uptake": "diagnostic_shadow"}


def _num(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _downstream_text(sem_trace):
    """The agent's terminal SURFACE (final answer / escalation) — the only place 'uptake' can show.
    Consumes ONLY SemanticEvent fields: we read terminal events' raw thought/content. No tool literals."""
    parts = []
    for s in (sem_trace or []):
        if s.get("terminal") in ("final", "escalate"):
            raw = s.get("raw") or {}
            ca = raw.get("canonical_action") if isinstance(raw, dict) else None
            for v in (raw.get("thought"), (ca or {}).get("content") if isinstance(ca, dict) else None):
                if v:
                    parts.append(str(v))
    return " ".join(parts).lower()


def _salient(payload):
    """Salient, matchable tokens of a delivered unit's payload: numbers(+optional unit) and content
    words >=4 chars. Generic/lexical — NO domain vocabulary list, so it stays benchmark-agnostic."""
    t = str(payload or "").lower()
    toks = set(re.findall(r"\d+(?:\.\d+)?\s?(?:cm|mm|hu|%|ml|mg|mmhg|kg|mcg)?", t))
    toks |= {w for w in re.findall(r"[a-z][a-z\-]{3,}", t)}
    return {x.strip() for x in toks if x.strip()}


# STRUCTURED tokens only (contract: NO long-word/lexical overlap for uptake). A typed identifier
# (MRN.../<Type>/<id>) or a numeric-with-UNIT atom (12mm / 35% / 8mg) -- specific values the agent
# RETRIEVED, whose reuse downstream is real evidence of uptake (bare small numbers/words are excluded).
_STRUCT_ID_RE = re.compile(r"(?:mrn\d+|[a-z][a-z]+/[\w.\-]+)", re.I)
_STRUCT_NUM_RE = re.compile(r"\d+(?:\.\d+)?\s?(?:cm|mm|hu|%|ml|mg|mcg|kg|mmhg|bpm|mmol|meq|iu)", re.I)


def _structured_tokens(payload):
    t = str(payload or "").lower()
    toks = {m.group(0).replace(" ", "") for m in _STRUCT_NUM_RE.finditer(t)}
    toks |= {x for x in _STRUCT_ID_RE.findall(t)}
    return {x for x in toks if len(x) >= 3}


def _authored_surface(sem_trace):
    """Everything the AGENT authored: its terminal decision text + every tool-call argument value. Used to
    detect whether a retrieved STRUCTURED value was actually reused (structured uptake)."""
    parts = [_downstream_text(sem_trace)]
    for s in (sem_trace or []):
        raw = s.get("raw") if isinstance(s.get("raw"), dict) else {}
        ca = raw.get("canonical_action") if isinstance(raw.get("canonical_action"), dict) else {}
        args = ca.get("args") if isinstance(ca.get("args"), dict) else (
            raw.get("args") if isinstance(raw.get("args"), dict) else {})
        for v in (args or {}).values():
            parts.append(str(v))
    return " ".join(parts).lower()


def _delivery_facts(u):
    """Derive the THREE distinct delivery facts for one evidence unit from the substrate fields
    {delivered_to_agent, error_visible}. These are independent — they do NOT contradict.
      result_evidence_delivered : the tool RESULT content reached the agent.
      error_signal_delivered    : a failure/error message reached the agent.
      observation_rendered      : SOMETHING (result OR error) was shown; False only if the renderer
                                  dropped everything.
    `error_bearing` marks a unit that represents a failure opportunity (the result did NOT get through,
    OR the unit explicitly flags an error) — the denominator for error_transparency."""
    result_delivered = bool(u.get("delivered_to_agent"))
    error_visible = bool(u.get("error_visible"))
    error_bearing = (u.get("delivered_to_agent") is False) or error_visible
    # the failure signal only counts as 'delivered' on a unit that actually carries a failure
    error_signal_delivered = bool(error_visible and error_bearing)
    observation_rendered = result_delivered or error_signal_delivered
    return {"observation_rendered": observation_rendered,
            "result_evidence_delivered": result_delivered,
            "error_signal_delivered": error_signal_delivered,
            "error_bearing": error_bearing}


def _layer(name, score, status, opportunities, basis):
    return {"name": name, "score": (round(score, 3) if score is not None else None),
            "status": status, "opportunities": opportunities, "basis": basis,
            "tier": _LAYER_TIER.get(name, _TIER)}



# ============================================================ Diagnostic Clinical Auditability (headline)
# OPERATOR-FACING. The question is NOT "did the agent get enough information" (that is Patient–Evidence Alignment) but "can an
# outside observer reconstruct what the system did, what failed, and how trustworthy the effects are".
#
#     O = mean_applicable(OperationalStateResolution, FailureDiagnosability,
#                         ReliabilityStateResolution, ResourceAccountability)
#
# Two design rules:
#
#  (a) FIELD PRESENT != OBSERVABLE. Every predicate must ask whether the telemetry RESOLVES the operator's
#      question, never whether a key exists. `status=failure, attribution=unknown` has every field yet is
#      barely diagnosable; `status=success, state_changed=unknown, no receipt` records the event yet leaves
#      the effect unknown. The decisive term is EFFECT RESOLUTION: status=partial means the trace itself
#      says the effect is unproven, so it must NOT score full marks.
#
#  (b) AGGREGATE BY FAMILY, THEN MACRO-AVERAGE. Scoring per span would make a 40-click run out-weigh a
#      10-click run on the same construct, i.e. introduce a trajectory-length confound. The
#      family partition comes from the plugin's observability_contract, so this scorer stays
#      benchmark-agnostic (same division of labour as Care Action Fulfillment's required_milestone_groups).
#
# ResourceAccountability is reported not_applicable unless the contract names real span-attributable cost
# signals. We do NOT synthesise a proxy just to fill the fourth term.
_TERMINALS = ("final", "escalate")
_TARGET_KEYS = ("ref", "selector", "target", "element", "element_id", "id", "resourceType", "file_path",
                "patient_reference", "subject", "name", "field", "url", "query", "image", "region",
                "expression", "direction")


def _dg_raw(s):
    return s.get("raw") or {}


def _dg_is_terminal(s):
    return s.get("terminal") in _TERMINALS and not _dg_raw(s).get("tool")


def _dg_target(e):
    a = e.get("args")
    if isinstance(a, dict):
        for k in _TARGET_KEYS:
            if a.get(k) not in (None, "", {}, []):
                return True
    return False


def _dg_result(e):
    for k in ("result", "observation", "canonical_result"):
        v = e.get(k)
        if (isinstance(v, dict) and v) or (isinstance(v, str) and v.strip()):
            return True
    return False


def _dg_families(sem_trace, contract):
    fam = {}
    fn = (contract or {}).get("family_of")
    for s in sem_trace or []:
        if callable(fn):
            try:
                k = fn(s)
            except Exception:
                k = "other"
        elif s.get("terminal") in _TERMINALS:
            k = "terminal"
        else:
            k = "cap:%s" % (s.get("capability_id") or s.get("event_role") or "unknown")
        fam.setdefault(k, []).append(s)
    return fam


def _dg_macro(per_family):
    vals = [v for v in per_family.values() if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _dg_class_token(e):
    """A failure is CLASSIFIED only when a recognisable class/code is present -- an explicit field, or a
    structured code in the payload. Free prose ("something went wrong") is not a classification."""
    for k in ("failure_mode", "failure_tag", "error_type", "error_class", "code"):
        if str(e.get(k) or "").strip():
            return True
    blob = ""
    for k in ("result", "observation", "canonical_result", "error"):
        v = e.get(k)
        blob += (v if isinstance(v, str) else str(v)) if v else ""
    return bool(re.search(r"http_\d{3}|\b[a-z]+_error\b|\b\d{3} client error\b|"
                          r"\b(?:TypeError|ValueError|KeyError|TimeoutError)\b", blob, re.I))


def _dg_result_resolved(e):
    """Is the RETURN of the operation interpretable BY AN OPERATOR -- not merely present in some field.

    Only the operator-facing channel counts: observation / canonical_observation / agent_visible_text
    (or a terminal's final_disposition). The machine fields `result` / `state_record` alone do NOT count:
    Care Action Fulfillment and Patient–Evidence Alignment already read those machine fields, so if Clinical Auditability accepted them, removing
    every operator-facing record would leave Clinical Auditability unchanged. A dimension whose whole claim is
    "an operator can tell what happened" has to read the operator-facing channel. All three benchmarks emit
    observation / canonical_observation / agent_visible_text on every tool_call, so this does not create a
    structural gap."""
    for k in ("observation", "canonical_observation", "agent_visible_text"):
        v = e.get(k)
        if (isinstance(v, dict) and v) or (isinstance(v, str) and v.strip()):
            return True
    return bool(str(e.get("final_disposition") or "").strip())


def _dg_effect_resolution(s):
    """Graded 1 / 0.5 / 0. A partial is NOT zero: the operator explicitly knows the effect is unconfirmed."""
    e = _dg_raw(s)
    if _dg_is_terminal(s):
        return 1.0 if str(e.get("final_disposition") or "").strip() else 0.0
    witnessed = bool(s.get("progress_token")) or bool(e.get("state_record")) or _dg_result_resolved(e)
    if s.get("status") == "partial":
        return 0.5 if witnessed or s.get("state_changed") is not None else 0.0
    if s.get("state_changed") is None:
        return 0.0
    return 1.0 if witnessed else 0.0


def _dg_reliability_resolution(s, receipt_fn=None):
    """How trustworthy is this recorded operation state? Single core -- no field-presence credit.

    THE RECEIPT MUST BE ENVIRONMENT-RETURNED, and only the plugin knows what that looks like. Generic
    proxies reduce to substrate-computed field presence: a bare field check, `state_changed is not None`
    (always computed), and `progress_token` (auto-generated by `_default_token` for any tool without a
    resolver). So the predicate is delegated to contract["receipt_resolver"]; with no resolver declared,
    R degrades to the result-legibility half-credit rather than inventing confirmation.
    """
    e = _dg_raw(s)
    if _dg_is_terminal(s):
        return 1.0 if str(e.get("final_disposition") or "").strip() else 0.0
    if callable(receipt_fn):
        try:
            receipt = bool(receipt_fn(s))
        except Exception:
            receipt = False
    else:
        receipt = False
    if s.get("event_role") == "commit":
        if receipt:
            return 1.0
        return 0.5 if (s.get("status") == "partial" and _dg_result_resolved(e)) else 0.0
    # read-only: structured result AND a real provenance/health signal -> 1; result only -> 0.5; label -> 0
    if _dg_result_resolved(e):
        return 1.0 if receipt else 0.5
    return 0.0


def diagnostic_observability(sem_trace, contract=None, raw_trace=None):
    """Diagnostic Clinical Auditability headline (version tag "v6-semantic-resolution"). Every atom is a SEMANTIC
    RESOLUTION predicate; capability_id / status / "a failure was selected" / "a terminal exists" are
    eligibility only and earn nothing."""
    fam = _dg_families(sem_trace, contract)
    if not fam:
        return {"score": None, "status": "not_applicable", "basis": "no semantic spans"}
    tr = (contract or {}).get("target_resolver")
    rc = (contract or {}).get("receipt_resolver")

    s_fam, r_fam, f_fam = {}, {}, {}
    order = list(sem_trace or [])
    for k, ss in fam.items():
        sv, rv, fv = [], [], []
        for s in ss:
            e = _dg_raw(s)
            term = _dg_is_terminal(s)
            # --- S: target / result / effect resolution (mean over the APPLICABLE ones)
            if term:
                t_res = 1.0 if str(e.get("final_disposition") or "").strip() else 0.0
            elif callable(tr):
                try:
                    t_res = 1.0 if tr(s) else 0.0
                except Exception:
                    t_res = 0.0
            else:
                t_res = None
            parts = [x for x in (t_res,
                                 1.0 if _dg_result_resolved(e) else 0.0,
                                 _dg_effect_resolution(s)) if x is not None]
            sv.append(sum(parts) / len(parts))
            # --- R
            rv.append(_dg_reliability_resolution(s, receipt_fn=rc))
            # --- F (failure spans only)
            if s.get("status") == "failure":
                classified = 1.0 if _dg_class_token(e) else 0.0
                attributed = 1.0 if s.get("failure_attribution") not in (None, "unknown") else 0.0
                ob = s.get("obligation_id")
                try:
                    i = order.index(s)
                except ValueError:
                    i = len(order)
                # causal linkage, not "any later success on the same capability"
                linked = bool(ob) and any(x.get("obligation_id") == ob and x.get("status") == "success"
                                          for x in order[i + 1:])
                linked = linked or bool(s.get("recovery_role")) or bool(e.get("recovery_of"))
                fv.append((classified + attributed + (1.0 if linked else 0.0)) / 3.0)
        s_fam[k] = (sum(sv) / len(sv)) if sv else None
        r_fam[k] = (sum(rv) / len(rv)) if rv else None
        if fv:
            f_fam[k] = sum(fv) / len(fv)

    S, R = _dg_macro(s_fam), _dg_macro(r_fam)
    F = _dg_macro(f_fam) if f_fam else None
    keys = list((contract or {}).get("cost_signals") or [])
    Cost = None
    if keys and raw_trace:
        seen = sum(1 for e in raw_trace
                   if isinstance(e, dict) and any(e.get(k) not in (None, "", {}, []) for k in keys))
        Cost = round(seen / len(raw_trace), 4)

    axes = {"operational_state_resolution": S, "failure_diagnosability": F,
            "reliability_state_resolution": R, "resource_accountability": Cost}
    applic = [v for v in axes.values() if v is not None]
    return {"score": round(sum(applic) / len(applic), 3) if applic else None,
            "status": "applicable" if applic else "not_applicable",
            "axes": axes, "applicable_axes": sorted(k for k, v in axes.items() if v is not None),
            "n_families": len(fam), "families": sorted(fam),
            "per_family": {"operational_state_resolution": s_fam,
                           "failure_diagnosability": f_fam,
                           "reliability_state_resolution": r_fam},
            "version": "v6-semantic-resolution",
            "measures": "operator-facing resolution of target, result, effect, failure and reliability",
            "note": "structural-presence predicates (capability_id / status / terminal existence) are "
                    "eligibility only and earn no credit; resource_accountability stays N/A until a plugin "
                    "declares span-attributable cost telemetry"}


def observability(evidence, sem_trace=None, actor_scope="agent", contract=None):
    """evidence: EvidenceView = [ {delivered_to_agent, delivery_fidelity, error_visible, acknowledged?,
    payload?} ]. sem_trace: optional SemanticTrace (only for the uptake fallback surface).

    Returns the layers SEPARATELY (delivery / fidelity / error_transparency / uptake — each its own
    number under `layers` and mirrored at top level) plus the applicable-only `composite`, and
    the per-unit `delivery_facts` rollup (observation_rendered / result_evidence_delivered /
    error_signal_delivered) so the three facts are not collapsed into one contradictory pair.
    The headline `score` is diagnostic_observability()'s score, falling back to the composite when
    that is not applicable."""
    # ACTOR-AWARE: agent-facing observability counts AGENT evidence only; a harness/recovery internal
    # read is auditability, not undelivered-to-agent. Pass actor_scope!="agent" for other profiles.
    if actor_scope == "agent" and evidence:
        _f = [u for u in evidence if (u or {}).get("actor", "agent") != "harness"]
        if _f:
            evidence = _f
        if sem_trace:
            _fs = [s for s in sem_trace if s.get("actor", "agent") != "harness"]
            if _fs:
                sem_trace = _fs

    units = list(evidence or [])
    n = len(units)
    if n == 0:
        return {"dimension": "Clinical Auditability", "tier": _TIER, "score": None, "composite": None,
                "status": "not_applicable", "coverage": 0.0, "reportable": False,
                "applicable_layers": [], "layers": {}, "submetrics": {}, "delivery_facts": {},
                "delivery": None, "fidelity": None, "error_transparency": None, "uptake": None,
                "basis": "no evidence units in trace"}

    facts = [_delivery_facts(u) for u in units]
    n_rendered = sum(1 for f in facts if f["observation_rendered"])
    n_result = sum(1 for f in facts if f["result_evidence_delivered"])
    err_idx = [i for i, f in enumerate(facts) if f["error_bearing"]]
    n_err = len(err_idx)
    n_err_signalled = sum(1 for i in err_idx if facts[i]["error_signal_delivered"])

    delivered_units = [units[i] for i, f in enumerate(facts) if f["result_evidence_delivered"]]
    nd = len(delivered_units)

    # -- delivery: the actual tool RESULT content that reached the agent (result_evidence_delivered,
    #    NOT 'no error'). --
    delivery = _layer("delivery", n_result / n, "applicable", n,
                      "%d/%d evidence units: result content reached the agent "
                      "(rendered-anything %d/%d)" % (n_result, n, n_rendered, n))

    # -- fidelity: of what WAS delivered, how intact (resolved localization / lossless) --
    if nd:
        fid = sum((_num(u.get("delivery_fidelity"), 1.0)) for u in delivered_units) / nd
        fidelity = _layer("fidelity", fid, "applicable", nd,
                          "mean fidelity over %d delivered units" % nd)
    else:
        fidelity = _layer("fidelity", None, "not_applicable", 0, "nothing delivered")

    # -- error_transparency: of the error-bearing units, how many surfaced the failure signal visibly.
    #    Denominator = error-bearing units (result did not get through OR explicit error flag); a unit
    #    that dropped the result AND dropped the error is the silent-failure case the layer penalises. --
    if n_err:
        error_transparency = _layer("error_transparency", n_err_signalled / n_err, "applicable", n_err,
                                    "%d/%d error-bearing units surfaced the failure signal visibly"
                                    % (n_err_signalled, n_err))
    else:
        error_transparency = _layer("error_transparency", None, "not_applicable", 0,
                                    "no error-bearing units")

    # -- realized uptake (EXPLICIT / STRUCTURED ONLY) -- the contract FORBIDS long-word/lexical
    #    overlap (it rewards copying, long answers, and accidental word coincidence). Applicable ONLY via an
    #    explicit substrate `acknowledged` flag, OR a STRUCTURED reuse: a typed id / numeric-with-unit atom
    #    the agent RETRIEVED that later appears in a tool-call argument or the terminal decision (the agent
    #    demonstrably USED that value). Otherwise not_applicable -- NO lexical fallback.
    ack_flags = [u for u in delivered_units if u.get("acknowledged") is not None]
    if ack_flags:
        up = sum(1 for u in ack_flags if u.get("acknowledged")) / len(ack_flags)
        uptake = _layer("uptake", up, "applicable", len(ack_flags),
                        "%d/%d delivered units acknowledged (substrate flag)"
                        % (sum(1 for u in ack_flags if u.get("acknowledged")), len(ack_flags)))
    else:
        surface = _authored_surface(sem_trace)
        surface_ns = surface.replace(" ", "")
        scored = [u for u in delivered_units if _structured_tokens(u.get("payload"))]
        if scored and surface:
            hit = 0
            for u in scored:
                toks = _structured_tokens(u.get("payload"))
                if any((tk in surface) or (tk in surface_ns) for tk in toks):
                    hit += 1
            uptake = _layer("uptake", hit / len(scored), "applicable", len(scored),
                            "%d/%d delivered units whose STRUCTURED value (typed id / numeric atom) the agent "
                            "reused in a later arg/decision" % (hit, len(scored)))
        else:
            uptake = _layer("uptake", None, "not_applicable", 0,
                            "no explicit acknowledgement and no structured value to trace (no lexical fallback)")

    # TWO LEVELS: system observability (env DELIVERY -- delivery/fidelity/error_transparency, the layers a
    # FIXED harness makes equal across models) and, ONLY when a structured/explicit signal exists, realized
    # uptake. Uptake is NEVER a fixed slice that silently drags the system score, and NEVER lexical. When
    # uptake is N/A, Clinical Auditability == system score (it MAY stay flat across models -- that is an honest
    # reading of an equal-delivery harness, not a bug).
    sys_layers = {l["name"]: l for l in (delivery, fidelity, error_transparency)}
    sys_applic = [l for l in sys_layers.values() if l["status"] == "applicable"]
    stw = sum(_SYS_W[l["name"]] for l in sys_applic)
    system_obs = round(sum(l["score"] * _SYS_W[l["name"]] for l in sys_applic) / stw, 3) if stw else None
    # Uptake is DIAGNOSTIC ONLY: the structured-uptake signal can invert against capability (it fires as 0.0
    # on ACTIVE retrievers that pull structured values but do not restate them -> penalizes strong agents,
    # while an agent that retrieves little gets uptake N/A -> no penalty). Correctness takes priority over
    # variance: do not force discrimination on an equal-delivery harness. So uptake is computed and reported
    # (below) but NOT folded into the composite; the composite == system observability
    # (delivery/fidelity/error_transparency).
    composite = system_obs
    layers = {l["name"]: l for l in (delivery, fidelity, error_transparency, uptake)}
    applicable = [l for l in layers.values() if l["status"] == "applicable"]
    out_system_obs = system_obs

    submetrics = {k: v["score"] for k, v in layers.items()}
    coverage = round(len(applicable) / len(layers), 3)
    delivery_facts = {
        "n_units": n,
        "observation_rendered": "%d/%d" % (n_rendered, n),
        "result_evidence_delivered": "%d/%d" % (n_result, n),
        "error_bearing_units": n_err,
        "error_signal_delivered": "%d/%d" % (n_err_signalled, n_err) if n_err else "0/0",
        "silent_failures": n_err - n_err_signalled,  # error-bearing units that surfaced NOTHING/no error
    }
    # ---- HEADLINE: Diagnostic Clinical Auditability. The delivery/fidelity/error_transparency/uptake
    # layers are KEPT and reported, but they answer an agent-facing question ("was evidence delivered into
    # the agent's context"). The ETCLOVG taxonomy (Li et al., Agent Harness Engineering: A Survey)
    # assigns these to Context, specialized here as Patient–Evidence Alignment. They are
    # diagnostics; the headline is operator-facing reconstructability.
    _dg = diagnostic_observability(sem_trace, contract=contract, raw_trace=None)
    _headline = _dg.get("score") if isinstance(_dg.get("score"), (int, float)) else composite
    return {"dimension": "Clinical Auditability", "tier": _TIER,
            "score": _headline, "composite": composite,
            "diagnostic_observability": _dg,
            "legacy_delivery_composite": composite,
            "status": "applicable" if composite is not None else "not_applicable",
            "coverage": coverage, "reportable": composite is not None,
            "applicable_layers": [l["name"] for l in applicable],
            "layers": layers, "submetrics": submetrics,
            # the layers, surfaced SEPARATELY at top level (each its own number, None if N/A):
            "system_observability": out_system_obs,
            "delivery": delivery["score"], "fidelity": fidelity["score"],
            "error_transparency": error_transparency["score"], "uptake": uptake["score"],
            "uptake_tier": "proxy",
            "delivery_facts": delivery_facts,
            "measures": "execution-system observability split into 3 delivery facts "
                        "(observation_rendered / result_evidence_delivered / error_signal_delivered) -> "
                        "layers delivery+fidelity+error_transparency+uptake(proxy); supersedes keyword "
                        "tool-type + lexical uptake",
            "basis": "; ".join(l["basis"] for l in layers.values())}


# --------------------------------------------------------------------------- self-verification
if __name__ == "__main__":
    import os, sys, json
    sys.path.insert(0, "runner")
    import substrate as S

    def _load_trace(p):
        return [json.loads(ln) for ln in open(p) if ln.strip()]

    def _run(label, bundle, benchmark):
        traj = os.path.join(bundle, "trajectory.jsonl")
        trace = _load_trace(traj)
        plugin = S.get_plugin(benchmark)
        ev = S.evidence_view(trace, plugin)
        sem = S.map_trace(trace, plugin)
        res = observability(ev, sem)
        print("\n===== %s (%s) =====" % (label, benchmark))
        print("  units=%d  composite=%s  status=%s  coverage=%s  reportable=%s"
              % (len(ev), res["composite"], res["status"], res["coverage"], res["reportable"]))
        df = res["delivery_facts"]
        print("  DELIVERY FACTS (3 distinct):  observation_rendered=%s  result_evidence_delivered=%s  "
              "error_signal_delivered=%s  silent_failures=%s"
              % (df["observation_rendered"], df["result_evidence_delivered"],
                 df["error_signal_delivered"], df["silent_failures"]))
        print("  LAYERS (separate):  delivery=%s  fidelity=%s  error_transparency=%s  uptake=%s (proxy)"
              % (res["delivery"], res["fidelity"], res["error_transparency"], res["uptake"]))
        for nm, l in res["layers"].items():
            print("    %-18s score=%-6s status=%-15s tier=%-12s opp=%s :: %s"
                  % (nm, l["score"], l["status"], l["tier"], l["opportunities"], l["basis"]))
        return res

    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _run("MedCTA",        os.path.join(base, "examples/medcta/MCTA-0"), "MedCTA")
    pb = os.path.join(base, "examples/pb/PB-adrenal_incidentaloma")
    if os.path.isdir(pb):
        _run("PhysicianBench", pb, "PhysicianBench")
    hab = os.path.join(base, "examples/hab/HAB-denial-hard-11")
    if os.path.isdir(hab):
        _run("HealthAdminBench", hab, "HealthAdminBench")
    print("\nimport OK")
