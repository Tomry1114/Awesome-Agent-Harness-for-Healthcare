"""Evidence FACETS — derive, from the PUBLIC question only, WHICH kinds of perceptual evidence a grounded
answer needs (localization / measurement / morphology / vascular relationship / temporal comparison), and
which facets a gathered perception evidence already provides. Deterministic keyword mapping — NO model
call, NO gold/reference — so Module B can nudge the agent EARLY and SPECIFICALLY ("you localized but never
assessed the vessel") instead of a single all-or-nothing judgement at the very end. General across any
perceptual environment; names no tool and no dataset.
"""
import json

FACET_KEYWORDS = {
    "anatomical_localization": ["where", "location", "located", "segment", "lobe", "region", "side",
                                "which part", "position", "anatomic", "site"],
    "measurement": ["size", "measure", "diameter", " cm", " mm", "dimension", "how large", "how big",
                    "volume", "largest", "longest"],
    "morphology": ["shape", "margin", "spiculat", "contour", "morpholog", "appearance", "characteriz",
                   "density", "enhanc", "calcif", "texture", "border"],
    "vascular_relationship": ["vascular", "vein", "artery", "vessel", "invasion", "invade", "thrombos",
                              "encasement", "encase", "portal", "aorta", "patency", "involv", "occlu"],
    "temporal_comparison": ["compare", "comparison", "prior", "previous", "change", "progress", "interval",
                            "since", "growth", "stable", "new lesion", "follow-up", "followup"],
}


def _blob(goal, context):
    return (str(goal or "") + " " + json.dumps(context or {}, ensure_ascii=False)).lower()


def required_facets(goal, context):
    """Facets the QUESTION asks about. Empty -> the question implies no specific facet (fall back to the
    generic image-grounding obligation; facets add nothing)."""
    t = _blob(goal, context)
    return sorted({f for f, kws in FACET_KEYWORDS.items() if any(k in t for k in kws)})


def evidence_facets(e):
    """Which facets a single PERCEPTION evidence record provides, from its (full) text + region args."""
    blob = (str(e.get("value_full") or "") + " " + str(e.get("value") or "") + " "
            + str(e.get("type") or "")).lower()
    return {f for f, kws in FACET_KEYWORDS.items() if any(k in blob for k in kws)}


def missing_facets(required, evidence):
    """Required facets not yet covered by any VALIDATED perception evidence."""
    covered = set()
    for e in (evidence or []):
        if (e.get("source_class") or e.get("source_type")) == "perception" and e.get("status") == "VALIDATED":
            covered |= evidence_facets(e)
    return [f for f in (required or []) if f not in covered]
