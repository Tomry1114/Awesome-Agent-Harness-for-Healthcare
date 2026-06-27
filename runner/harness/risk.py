"""Action risk tiers (R0-R3) — derived from the DECLARED action effect, not keyword guessing.

  R0 read         effect=none          -> allow + record
  R1 reversible   effect=reversible    -> prospective check
  R2 commit       effect=irreversible  -> prospective + retrospective verification
  R3 unjudgeable  declared R3           -> ESCALATE

The effect is declared by the substrate manifest (semantics.canonicalize). A contract commit point may
override the tier. No tool-name / keyword heuristics live here.
"""
from .semantics import EFFECT_RISK

R0, R1, R2, R3 = "R0", "R1", "R2", "R3"
ORDER = {R0: 0, R1: 1, R2: 2, R3: 3}


def classify_risk(sem, contract=None):
    """sem = SemanticAction. A matching contract commit point's declared risk wins; else effect -> tier."""
    if contract is not None:
        cp = contract.commit_point_for(sem)
        if cp and cp.get("risk"):
            return cp["risk"]
    return EFFECT_RISK.get(getattr(sem, "effect", "none"), R0)


def at_least(risk, threshold):
    return ORDER.get(risk, 0) >= ORDER.get(threshold, 0)
