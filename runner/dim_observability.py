#!/usr/bin/env python3
"""Observability dimension (experimental tier) — benchmark-AGNOSTIC.

Question: did the EXECUTION SYSTEM make the world observable to the deciding agent? i.e. was task
evidence and were failures actually DELIVERED, at fidelity, into the agent's context — and were they
visible / taken up downstream. This SUPERSEDES proxy_verifiers.observability(), which guessed tool
TYPE from tool-name keywords ("search"/"read"/"ocr"...) and measured uptake by lexical long-word
overlap. Here we consume ONLY the substrate EvidenceView fields the plugin already populated:
    delivered_to_agent, delivery_fidelity, error_visible, acknowledged (optional)
plus the SemanticTrace (only to find the downstream agent surface for an optional uptake fallback).
NO benchmark name / tool literal / image / DOM / FHIR string appears in the scoring logic.

THREE DISTINCT DELIVERY FACTS (review fix) — a single 0.74 hid the layers, and the old code set the
contradictory pair delivered_to_agent=False AND error_visible=True on the SAME unit. That pair is NOT
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

Entry: observability(evidence, sem_trace) -> {score, status, coverage, reportable, layers, delivery_facts,
delivery / fidelity / error_transparency / uptake (each its own number), composite, submetrics...}
"""
import re

_W = {"delivery": 0.4, "fidelity": 0.3, "error_transparency": 0.2, "uptake": 0.1}
_TIER = "experimental"
# uptake is a weaker, lexical/heuristic signal — flagged proxy so the composite consumer can see it.
_LAYER_TIER = {"delivery": "experimental", "fidelity": "experimental",
               "error_transparency": "experimental", "uptake": "proxy"}


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


def observability(evidence, sem_trace=None):
    """evidence: EvidenceView = [ {delivered_to_agent, delivery_fidelity, error_visible, acknowledged?,
    payload?} ]. sem_trace: optional SemanticTrace (only for the uptake fallback surface).

    Returns the layers SEPARATELY (delivery / fidelity / error_transparency / uptake — each its own
    number under `layers` and mirrored at top level) plus the applicable-only `composite`/`score`, and
    the per-unit `delivery_facts` rollup (observation_rendered / result_evidence_delivered /
    error_signal_delivered) so the three facts are no longer collapsed into one contradictory pair."""
    units = list(evidence or [])
    n = len(units)
    if n == 0:
        return {"dimension": "Observability", "tier": _TIER, "score": None, "composite": None,
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

    # -- delivery: the actual tool RESULT content that reached the agent (review: use
    #    result_evidence_delivered, NOT 'no error'). --
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

    # -- uptake (PROXY, optional): delivered evidence referenced/acknowledged downstream --
    # Prefer an explicit substrate `acknowledged` flag; else fall back to matching salient payload
    # tokens against the agent's terminal surface. N/A when neither signal is available.
    ack_flags = [u for u in delivered_units if u.get("acknowledged") is not None]
    if ack_flags:
        up = sum(1 for u in ack_flags if u.get("acknowledged")) / len(ack_flags)
        uptake = _layer("uptake", up, "applicable", len(ack_flags),
                        "%d/%d delivered units acknowledged (substrate flag)"
                        % (sum(1 for u in ack_flags if u.get("acknowledged")), len(ack_flags)))
    else:
        surface = _downstream_text(sem_trace)
        scored = [u for u in delivered_units if _salient(u.get("payload"))]
        if surface and scored:
            hit = 0
            for u in scored:
                toks = _salient(u.get("payload"))
                if toks and any(tk in surface for tk in toks):
                    hit += 1
            uptake = _layer("uptake", hit / len(scored), "applicable", len(scored),
                            "%d/%d delivered units referenced in terminal surface (token fallback, proxy)"
                            % (hit, len(scored)))
        else:
            uptake = _layer("uptake", None, "not_applicable", 0,
                            "no downstream surface / no matchable payload")

    layers = {l["name"]: l for l in (delivery, fidelity, error_transparency, uptake)}
    applicable = [l for l in layers.values() if l["status"] == "applicable"]
    tw = sum(_W[l["name"]] for l in applicable)
    composite = round(sum(l["score"] * _W[l["name"]] for l in applicable) / tw, 3) if tw else None

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
    return {"dimension": "Observability", "tier": _TIER,
            "score": composite, "composite": composite,
            "status": "applicable" if composite is not None else "not_applicable",
            "coverage": coverage, "reportable": composite is not None,
            "applicable_layers": [l["name"] for l in applicable],
            "layers": layers, "submetrics": submetrics,
            # the layers, surfaced SEPARATELY at top level (each its own number, None if N/A):
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

    base = os.path.expanduser("~/Medical_harness")
    _run("MedCTA",        os.path.join(base, "results_mctaGov/gpt5/MCTA-0"), "MedCTA")
    pb = os.path.join(base, "results_pb_chk3/gpt5/PB-aberrant_drug_screen")
    if os.path.isdir(pb):
        _run("PhysicianBench", pb, "PhysicianBench")
    hab = os.path.join(base, "results_hab10/gpt5/HAB-denial-easy-1")
    if os.path.isdir(hab):
        _run("HealthAdminBench", hab, "HealthAdminBench")
    print("\nimport OK")
