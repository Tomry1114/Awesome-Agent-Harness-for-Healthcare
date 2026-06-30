"""ACQUIRE capability (Evidence-Driven Harness, minimal slice). Instead of rewriting the answer with the SAME
evidence (reactive repair -> can't fix capability errors), the harness DETECTS a recoverable gap and ACQUIRES
NEW read-only evidence: it elicits the agent's two most plausible interpretations + the single discriminating
observable, and if that target was never focus-observed, asks the kernel/runner to acquire it. The agent then
re-reasons with strictly MORE information than its first pass -> a genuine first-pass<achievable gap the
harness can close WITHOUT a stronger model and WITHOUT gold. Perception substrates only; capped per task."""
import os
from ..capability import Capability
from .. import decision as D
from .. import affordance

def _enabled():
    return os.environ.get("MH_REPAIR", "hard") in ("soft", "select", "full")


class ActiveEvidence(Capability):
    # LAYER: AMPLIFICATION (information acquisition) -- adds capability via read-only evidence, never decides the answer
    name = "active_evidence"

    def before_final(self, answer, ctx):
        if not _enabled() or not ctx.judge_fn or not getattr(ctx, "final_is_commit", False):
            return None
        meta = (ctx.contract.meta or {}) if ctx.contract else {}
        tools = meta.get("available_tools") or []
        sel0 = affordance.select_tools(tools)
        if not sel0:
            return None                                    # no perception affordance -> not applicable
        led = ctx.ledger
        if getattr(led, "acquire_count", 0) >= 2:
            return None                                    # per-task acquisition cap (anti tool-spam)
        if not ctx.spend_semantic():
            return None
        from ..engines.semantic import elicit_discriminator
        obs_summ = []
        for o in getattr(led, "observations", []):
            if o.get("result_status") in ("valid", "ok") and (o.get("content") or o.get("region")):
                r, c = o.get("region"), (o.get("content") or "")
                obs_summ.append((("%s: %s" % (r, c)) if r else str(c))[:200])
        disc = elicit_discriminator(answer, meta.get("goal") or meta.get("public_context"),
                                    observations=obs_summ, judge_fn=ctx.judge_fn)
        if not disc or not disc.get("region"):
            return None                                    # certain / non-perceptual -> nothing to acquire
        region, attribute = disc.get("region"), disc.get("attribute")
        from ..observation import _n
        observed = any(o.get("result_status") == "valid" and o.get("region") and _n(o.get("region")) == _n(region)
                       for o in getattr(led, "observations", []))
        if observed:
            return None                                    # the discriminator was already focus-observed
        sel = affordance.select_tools(tools, region=region) or sel0
        na = {"capability": "inspect_region_attribute", "tool": sel[0],
              "args": {"region": region, "attribute": attribute}, "read_only": True}
        return self._decide(D.ACQUIRE, rule_id="active_evidence", reason_code="unresolved_discriminator",
                            reason="acquire the discriminating observation (%s / %s) before committing the answer"
                                   % (region, attribute),
                            extra={"next_action": na, "discriminator": disc})
