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

Layers (applicable-only — a layer with no opportunity is status=not_applicable and EXCLUDED from the
mean, so a clean run never collects a vacuous 1.0):
  exposure           = delivered / total evidence units                     (always applies if units>0)
  delivery_fidelity  = mean fidelity over DELIVERED units                   (applies if any delivered)
  error_transparency = visible-errors / error-bearing units                (N/A if no errors)
  uptake (optional)  = delivered units referenced/acknowledged downstream   (N/A if no downstream text
                       AND no unit carries an `acknowledged` flag)

Entry: observability(evidence, sem_trace) -> {score, status, coverage, reportable, layers, submetrics...}
"""
import re

_W = {"exposure": 0.4, "delivery_fidelity": 0.3, "error_transparency": 0.2, "uptake": 0.1}
_TIER = "experimental"


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


def _layer(name, score, status, opportunities, basis):
    return {"name": name, "score": (round(score, 3) if score is not None else None),
            "status": status, "opportunities": opportunities, "basis": basis}


def observability(evidence, sem_trace=None):
    """evidence: EvidenceView = [ {delivered_to_agent, delivery_fidelity, error_visible, acknowledged?,
    payload?} ]. sem_trace: optional SemanticTrace (only for the uptake fallback surface)."""
    units = list(evidence or [])
    n = len(units)
    if n == 0:
        return {"dimension": "Observability", "tier": _TIER, "score": None,
                "status": "not_applicable", "coverage": 0.0, "reportable": False,
                "applicable_layers": [], "layers": {}, "submetrics": {},
                "basis": "no evidence units in trace"}

    delivered = [u for u in units if u.get("delivered_to_agent")]
    nd = len(delivered)

    # -- exposure: harness-side delivery of produced evidence into the agent's context --
    exposure = _layer("exposure", nd / n, "applicable", n,
                      "%d/%d evidence units delivered to agent" % (nd, n))

    # -- delivery_fidelity: of what WAS delivered, how intact (resolved localization / lossless) --
    if nd:
        fid = sum((_num(u.get("delivery_fidelity"), 1.0)) for u in delivered) / nd
        fidelity = _layer("delivery_fidelity", fid, "applicable", nd,
                          "mean fidelity over %d delivered units" % nd)
    else:
        fidelity = _layer("delivery_fidelity", None, "not_applicable", 0, "nothing delivered")

    # -- error_transparency: of the units that carry an error, how many surfaced it visibly --
    err_units = [u for u in units if (u.get("error_visible") is not None
                 and (u.get("error_visible") or u.get("delivered_to_agent") is False))]
    # An error opportunity = a unit that failed delivery OR explicitly flags an error state.
    err_bearing = [u for u in units if (u.get("delivered_to_agent") is False) or u.get("error_visible")]
    if err_bearing:
        visible = sum(1 for u in err_bearing if u.get("error_visible"))
        error_transparency = _layer("error_transparency", visible / len(err_bearing), "applicable",
                                    len(err_bearing),
                                    "%d/%d error-bearing units surfaced visibly" % (visible, len(err_bearing)))
    else:
        error_transparency = _layer("error_transparency", None, "not_applicable", 0,
                                    "no errors occurred")

    # -- uptake (optional): delivered evidence referenced/acknowledged downstream --
    # Prefer an explicit substrate `acknowledged` flag; else fall back to matching salient payload
    # tokens against the agent's terminal surface. N/A when neither signal is available.
    ack_flags = [u for u in delivered if u.get("acknowledged") is not None]
    if ack_flags:
        up = sum(1 for u in ack_flags if u.get("acknowledged")) / len(ack_flags)
        uptake = _layer("uptake", up, "applicable", len(ack_flags),
                        "%d/%d delivered units acknowledged (substrate flag)"
                        % (sum(1 for u in ack_flags if u.get("acknowledged")), len(ack_flags)))
    else:
        surface = _downstream_text(sem_trace)
        scored = [u for u in delivered if _salient(u.get("payload"))]
        if surface and scored:
            hit = 0
            for u in scored:
                toks = _salient(u.get("payload"))
                if toks and any(tk in surface for tk in toks):
                    hit += 1
            uptake = _layer("uptake", hit / len(scored), "applicable", len(scored),
                            "%d/%d delivered units referenced in terminal surface (token fallback)"
                            % (hit, len(scored)))
        else:
            uptake = _layer("uptake", None, "not_applicable", 0,
                            "no downstream surface / no matchable payload")

    layers = {l["name"]: l for l in (exposure, fidelity, error_transparency, uptake)}
    applicable = [l for l in layers.values() if l["status"] == "applicable"]
    tw = sum(_W[l["name"]] for l in applicable)
    score = round(sum(l["score"] * _W[l["name"]] for l in applicable) / tw, 3) if tw else None

    submetrics = {k: v["score"] for k, v in layers.items()}
    coverage = round(len(applicable) / len(layers), 3)
    return {"dimension": "Observability", "tier": _TIER,
            "score": score,
            "status": "applicable" if score is not None else "not_applicable",
            "coverage": coverage, "reportable": score is not None,
            "applicable_layers": [l["name"] for l in applicable],
            "layers": layers, "submetrics": submetrics,
            "measures": "execution-system observability (delivery+fidelity+error_visibility+uptake), "
                        "from EvidenceView fields — supersedes keyword tool-type + lexical uptake",
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
        print("  units=%d  score=%s  status=%s  coverage=%s  reportable=%s"
              % (len(ev), res["score"], res["status"], res["coverage"], res["reportable"]))
        print("  applicable_layers:", res["applicable_layers"])
        for nm, l in res["layers"].items():
            print("    %-18s score=%-6s status=%-15s opp=%s :: %s"
                  % (nm, l["score"], l["status"], l["opportunities"], l["basis"]))
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
