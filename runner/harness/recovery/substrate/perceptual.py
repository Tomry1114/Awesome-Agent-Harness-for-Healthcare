"""Bounded Clinical Recovery v3 - Perceptual Substrate Adapter (environment mechanics only).

HOW to observe a clinical image in the tool-sandbox environment, and NOTHING else. This adapter performs
READ-ONLY perceptual acquisition (ImageDescription / RegionAttributeDescription / OCR) through the injected
ToolSandboxEnv-style env (env.call_tool(name, args) -> {tool, args, output, ...}; env.available_tools()).

Four-layer separation (enforced):
  * It knows perceptual MECHANICS only: how a region tool reports localization, how an OCR read reports an
    empty page, how to select a perception affordance from the available tools. It resolves the perception
    tool by the affordance registry (region/attribute-capable tools first; web/search tools excluded there).
  * It knows NOTHING about answer-slots, findings, diagnoses, uncovered claims, or which benchmark it serves.
    The region it is asked to look at is provided by the caller (the workflow, derived from the public
    question); this adapter never derives a target from anything but the target_spec it is handed.
  * It NEVER mutates: no MutationAuthorization is ever consumed, no write/commit primitive is honored. A
    mutation kind is refused. read_state exposes the read-only evidence-ledger view.

classify_result maps a perceptual result to the kernel outcome vocabulary using the same two mechanics the
result-conditional perception resolvers use: a region read whose localization.resolved is False (a silent
whole-image fallback -> the targeted region was NOT examined) and an empty / no-text OCR page both yield
RESULT_UNKNOWN (evidence not actually obtained); a real localized/non-empty read yields RESULT_OK.

Python 3.8 compatible.
"""
import hashlib

from ..contracts import (
    AffordanceBinding, Outcome,
    RESULT_OK, RESULT_UNKNOWN, RESULT_FAILED, RESULT_ALREADY_REALIZED,
    BLOCKED_AMBIGUOUS_TARGET, BLOCKED_UNRESOLVED_AFFORDANCE,
    STAGED_WRITE, IRREVERSIBLE_COMMIT,
)
from ...affordance import select_tools


# OCR no-readable-text sentinels (perceptual mechanic; identical to the reader's documented [no text] sentinel).
_NO_TEXT = frozenset(("[no text]", "no text", ""))


def _hash(obj):
    try:
        return hashlib.sha1(str(obj).encode("utf-8")).hexdigest()[:12]
    except Exception:
        return ""


class PerceptualSubstrateAdapter(object):
    """SubstrateAdapter for read-only perceptual acquisition. Inject a ToolSandboxEnv-style env.

    env must expose:
        call_tool(name, args) -> {"tool":.., "args":.., "output": <str|dict>, ...}
        available_tools()     -> list[dict|str]   (tool descriptors with 'name'/'signature')
    """

    def __init__(self, env, available_tools=None, ledger=None):
        self._env = env
        self._available = available_tools           # optional explicit tool descriptors (else env.available_tools())
        self._ledger = list(ledger or [])           # acquired observations = the evidence ledger

    # -- SubstrateAdapter surface -----------------------------------------------------------------
    def resolve_affordance(self, target_spec, observation):
        """Pick a READ affordance (a perception tool) for a perceptual target. The target's region comes
        from the caller (public question), never from this adapter. A null/blank region cannot be uniquely
        located -> BLOCKED_AMBIGUOUS_TARGET; a region for which the environment exposes no perception tool
        -> BLOCKED_UNRESOLVED_AFFORDANCE; otherwise the region/attribute-capable tool ranked first."""
        spec = target_spec if isinstance(target_spec, dict) else {"region": target_spec}
        region = spec.get("region")
        modality = spec.get("modality")
        if region is None or (isinstance(region, str) and not region.strip()):
            return BLOCKED_AMBIGUOUS_TARGET
        tools = select_tools(self._tools(observation), region=region, modality=modality)
        if not tools:
            return BLOCKED_UNRESOLVED_AFFORDANCE
        return AffordanceBinding(target_spec=spec, ref=tools[0], observation_hash=_hash(observation))

    def execute_primitive(self, kind, action, auth):
        """Execute one READ-ONLY perceptual primitive. Read-like kinds must arrive with auth=None; a mutation
        kind is refused (this substrate performs no writes). A tool-backed acquire runs the selected perception
        tool and appends the result to the evidence ledger; an affordance-less acquire returns the ledger view
        (evidence-sufficiency confirmation) so the kernel can gate the next step on 'evidence_acquired'."""
        if kind in (STAGED_WRITE, IRREVERSIBLE_COMMIT):
            return Outcome(status=RESULT_FAILED, reason="perceptual_substrate_is_read_only")
        action = action or {}
        aff = action.get("affordance")
        tool = getattr(aff, "ref", None) if aff is not None else None
        inner = action.get("action") or {}
        spec = getattr(aff, "target_spec", None) if aff is not None else None
        spec = spec if isinstance(spec, dict) else {}

        if not tool:
            # evidence-sufficiency confirmation read: expose the ledger, derive no new evidence.
            usable = self._ledger_has_usable()
            sv = {"evidence_ledger": list(self._ledger)}
            if usable:
                sv["evidence_acquired"] = True
            return Outcome(status=RESULT_OK, result={"evidence_ledger": list(self._ledger)}, state_view=sv)

        region = inner.get("region") if inner.get("region") is not None else spec.get("region")
        attribute = inner.get("attribute") if inner.get("attribute") is not None else spec.get("attribute")
        modality = inner.get("modality") if inner.get("modality") is not None else spec.get("modality")
        args = {}
        if region is not None:
            args["region_query"] = region
        if attribute is not None:
            args["attribute"] = attribute
        try:
            envelope = self._env.call_tool(tool, args)
        except Exception as ex:
            return Outcome(status=RESULT_FAILED, reason="tool_error:%r" % (ex,))
        output = envelope.get("output") if isinstance(envelope, dict) else envelope
        resolved = self._localized(output)
        empty = self._empty(output)
        usable = (not empty) and (resolved is not False)

        obs = {
            "observation_id": "obs-%d" % len(self._ledger),
            "tool_capability": tool,
            "region": region, "modality": modality, "attribute": attribute,
            "attributes_observed": [attribute] if attribute else [],
            "result_status": "valid" if usable else "empty",
            "content": self._text(output),
            "source_channel": "radiology_image", "source_instance_id": "image:primary",
            "extractor": "OCR" if str(tool).upper() == "OCR" else "image_vlm",
            "localization_resolved": resolved,
        }
        self._ledger.append(obs)
        sv = {"evidence_ledger": list(self._ledger)}
        if resolved is True:
            sv["region_localized"] = True
        if usable:
            sv["evidence_acquired"] = True
        return Outcome(status=RESULT_OK, result=output, state_view=sv,
                       raw={"localization_resolved": resolved, "empty": empty, "usable": usable})

    def read_state(self, paths):
        """Authoritative read-back of the read-only evidence ledger view (no server round-trip / no write)."""
        view = {"evidence_ledger": list(self._ledger)}
        if self._ledger_has_usable():
            view["evidence_acquired"] = True
        if any(o.get("localization_resolved") is True for o in self._ledger):
            view["region_localized"] = True
        return view

    def classify_result(self, result):
        """RESULT_OK / RESULT_UNKNOWN / RESULT_FAILED / RESULT_ALREADY_REALIZED for a perceptual result.
        A region read that fell back to the whole image (localization.resolved False) and an empty / no-text
        OCR page both mean the evidence was NOT actually obtained -> RESULT_UNKNOWN."""
        status = getattr(result, "status", None)
        if status == RESULT_FAILED:
            return RESULT_FAILED
        if status == RESULT_ALREADY_REALIZED:
            return RESULT_ALREADY_REALIZED
        output = getattr(result, "result", result)
        if self._localized(output) is False:
            return RESULT_UNKNOWN
        if self._empty(output):
            return RESULT_UNKNOWN
        return RESULT_OK

    # -- perceptual mechanics ---------------------------------------------------------------------
    def _tools(self, observation):
        if isinstance(observation, dict) and observation.get("available_tools") is not None:
            return observation.get("available_tools")
        if self._available is not None:
            return self._available
        try:
            return self._env.available_tools()
        except Exception:
            return []

    @staticmethod
    def _localized(output):
        """True/False when the output carries an explicit region-localization status, else None."""
        if isinstance(output, dict):
            loc = output.get("localization")
            if isinstance(loc, dict) and "resolved" in loc:
                return bool(loc.get("resolved"))
        return None

    def _empty(self, output):
        txt = self._text(output)
        return txt.strip().lower() in _NO_TEXT

    @staticmethod
    def _text(output):
        if isinstance(output, dict):
            return str(output.get("text") or output.get("snippet") or "")
        return str(output or "")

    def _ledger_has_usable(self):
        return any(o.get("result_status") == "valid" for o in self._ledger)
