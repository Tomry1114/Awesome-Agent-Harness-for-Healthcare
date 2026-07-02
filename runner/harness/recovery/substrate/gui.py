"""Bounded Clinical Recovery v3 - GUI substrate adapter (SubstrateAdapter).

Environment mechanics ONLY. This module knows how to observe and act in the browser portal: parse the
live observation into interactive-element refs, locate a control uniquely, click/type/upload/submit via
the driver's tool surface, and read the authoritative persisted state back. It knows NOTHING about any
clinical/admin process (no prior-auth, no appeal, no decision-documentation) and NOTHING about any
benchmark's task fields. Reversibility is declared by the caller's per-step manifest, not inferred here.

Design refs: v3 sec.2 (SubstrateAdapter), sec.5 (affordance terminals / tiered auth), sec.7 (GUI
snapshot/resolve/click/type/upload + read payer state).

Oracle-blind: reads only the LIVE observation and the environment's own persisted localStorage; never
gold/checkpoint/reference material. Python 3.8 compatible.
"""
import hashlib
import json
import re

from ..contracts import (
    AffordanceBinding, Outcome,
    NAVIGATE, STAGED_WRITE, IRREVERSIBLE_COMMIT,
    READ_LIKE_KINDS,
    RESULT_OK, RESULT_UNKNOWN, RESULT_FAILED, RESULT_ALREADY_REALIZED,
    BLOCKED_UNRESOLVED_AFFORDANCE, BLOCKED_AMBIGUOUS_TARGET,
)


# --------------------------------------------------------------------------------------------------
# Observation parsing + the pure DOM-ref affordance resolver (module level so tests and the stub can
# exercise the REAL logic without a browser).
# --------------------------------------------------------------------------------------------------
# Element line emitted by the portal observer: "[ref=N] <tag>[<type>] '<label>'( value='<v>')?"
_ELEM_RE = re.compile(r"^\[ref=(\d+)\]\s+([A-Za-z0-9]+)(?:\[([^\]]*)\])?\s+(.*)$")

# role -> the set of tags that realize it (a control also matches if its declared type/role == role).
_ROLE_TAGS = {
    "button": {"button"},
    "submit": {"button"},
    "link": {"a"},
    "tab": {"a", "button"},
    "menuitem": {"a", "button"},
    "input": {"input", "textarea", "select"},
    "textbox": {"input", "textarea"},
    "textarea": {"textarea"},
    "select": {"select"},
    "combobox": {"select", "input", "button"},
    "checkbox": {"input"},
    "radio": {"input"},
    "switch": {"input"},
}

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")


def _norm_label(text):
    """Lower-case, drop punctuation, collapse whitespace: a stable label key for matching."""
    if text is None:
        return ""
    s = str(text).lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def _strip_label(rest):
    """Pull the human label out of the trailing '<label>'( value=...)? portion of an element line."""
    # drop a trailing " value=..." suffix first (labels never contain that token verbatim here)
    cut = rest.split(" value=", 1)[0].strip()
    if len(cut) >= 2 and cut[0] in "'\"" and cut[-1] == cut[0]:
        cut = cut[1:-1]
    return cut


def parse_elements(observation):
    """Parse an observation string into a list of {ref, tag, type, label, norm} dicts."""
    els = []
    if not observation:
        return els
    for raw_line in str(observation).splitlines():
        line = raw_line.strip()
        m = _ELEM_RE.match(line)
        if not m:
            continue
        ref = int(m.group(1))
        tag = (m.group(2) or "").lower()
        etype = (m.group(3) or "").lower()
        label = _strip_label(m.group(4) or "")
        els.append({"ref": ref, "tag": tag, "type": etype,
                    "label": label, "norm": _norm_label(label)})
    return els


def _role_of(spec):
    if isinstance(spec, dict):
        return (spec.get("role") or spec.get("kind") or "") or None
    return None


def _label_of(spec):
    if isinstance(spec, dict):
        return spec.get("label") or spec.get("text") or spec.get("name")
    return spec  # a bare string target_spec == the label


def _bound_id_of(spec):
    if isinstance(spec, dict):
        for k in ("bound_id", "ref", "disambiguator", "id"):
            if spec.get(k) not in (None, ""):
                return spec.get(k)
    return None


def _role_ok(role, el):
    if not role:
        return True
    role = str(role).lower()
    if role in (el["tag"], el["type"]):
        return True
    return el["tag"] in _ROLE_TAGS.get(role, {role})


def _label_ok(target_norm, el):
    if not target_norm:
        return True                       # role-only target
    en = el["norm"]
    if not en:
        return False
    return target_norm in en or en in target_norm


def _obs_hash(observation):
    return hashlib.md5(("" if observation is None else str(observation)).encode(
        "utf-8", "ignore")).hexdigest()[:12]


def resolve_affordance_in_observation(target_spec, observation):
    """The REAL live-observation DOM-ref resolver.

    Filter interactive elements by role, normalize+match labels, require EXACTLY ONE survivor:
      1 survivor              -> AffordanceBinding(target_spec, ref, observation_hash)
      0 survivors             -> BLOCKED_UNRESOLVED_AFFORDANCE
      >1 survivors            -> a disambiguating bound id that selects exactly one wins;
                                 otherwise BLOCKED_AMBIGUOUS_TARGET
    """
    els = parse_elements(observation)
    role = _role_of(target_spec)
    target_norm = _norm_label(_label_of(target_spec))
    survivors = [el for el in els if _role_ok(role, el) and _label_ok(target_norm, el)]

    if len(survivors) == 1:
        return AffordanceBinding(target_spec=target_spec, ref=survivors[0]["ref"],
                                 observation_hash=_obs_hash(observation))
    if len(survivors) == 0:
        return BLOCKED_UNRESOLVED_AFFORDANCE

    # >1 survivor: only a bound id (an exact ref, or a label discriminator) may break the tie.
    bid = _bound_id_of(target_spec)
    if bid is not None:
        bid_norm = _norm_label(bid)
        picked = [el for el in survivors
                  if str(el["ref"]) == str(bid) or (bid_norm and bid_norm == el["norm"])]
        if len(picked) == 1:
            return AffordanceBinding(target_spec=target_spec, ref=picked[0]["ref"],
                                     observation_hash=_obs_hash(observation))
    return BLOCKED_AMBIGUOUS_TARGET


# --------------------------------------------------------------------------------------------------
# GuiSubstrateAdapter - wraps the live GUI driver (a GuiEnvReal instance) and exposes the four
# SubstrateAdapter methods. No workflow / benchmark knowledge.
# --------------------------------------------------------------------------------------------------
class GuiSubstrateAdapter(object):
    """SubstrateAdapter over the real Playwright portal driver.

    `driver` is expected to expose the GuiEnvReal surface: .call_tool(name, args), .page, .full_state,
    and (optionally, once environments.py is extended) .read_recovery_state(). Every method degrades
    gracefully if a hook is missing so this adapter can be unit-tested with a stub.
    """

    def __init__(self, driver=None):
        self.driver = driver

    # -- affordance -------------------------------------------------------------------------------
    def resolve_affordance(self, target_spec, observation):
        return resolve_affordance_in_observation(target_spec, observation)

    # -- primitives -------------------------------------------------------------------------------
    def execute_primitive(self, kind, action, auth):
        act = (action or {}).get("action") or {}
        op = act.get("op") or kind
        aff = (action or {}).get("affordance")
        ref = getattr(aff, "ref", None)
        driver = self.driver
        if driver is None:
            return Outcome(status=RESULT_FAILED, reason="no_driver")

        try:
            if kind in READ_LIKE_KINDS:
                if op == "navigate" or kind == NAVIGATE:
                    r = driver.call_tool("navigate", {k: v for k, v in act.items() if k != "op"})
                else:
                    r = driver.call_tool("snapshot", {})
                return self._outcome_from(r)

            # mutation kinds: a single-use authorization must be present and consumable.
            if auth is None or not auth.consume():
                return Outcome(status=RESULT_FAILED, reason="unauthorized_mutation:%s" % kind)

            args = {}
            if ref is not None:
                args["ref"] = ref

            if kind == STAGED_WRITE:
                sub = op
                if sub in ("type", "fill"):
                    arg_name = act.get("arg")
                    val = (action.get("bindings") or {}).get(arg_name)
                    args["text"] = "" if val is None else str(val)
                    r = driver.call_tool("type", args)
                elif sub == "select":
                    arg_name = act.get("arg")
                    val = (action.get("bindings") or {}).get(arg_name)
                    args["value"] = "" if val is None else str(val)
                    r = driver.call_tool("select", args)
                elif sub == "upload":
                    args["file_ref"] = act.get("file_ref", "last")
                    r = driver.call_tool("upload", args)
                else:                                   # click / open / navigate-within
                    r = driver.call_tool("click", args)
                return self._outcome_from(r)

            if kind == IRREVERSIBLE_COMMIT:
                r = driver.call_tool("submit", args)
                out = self._outcome_from(r, commit=True)
                # active read-back reconciliation hook: a submit that left the persisted state
                # unchanged is UNKNOWN, never silently OK.
                rec = self._reconcile(driver, r)
                if rec is False:
                    out.status = RESULT_UNKNOWN
                    out.reason = "submit_unconfirmed_on_readback"
                return out

            return Outcome(status=RESULT_FAILED, reason="unknown_kind:%s" % kind)
        except Exception as e:                          # pragma: no cover - real-browser failure path
            return Outcome(status=RESULT_FAILED, reason="primitive_exception:%r" % (e,))

    def _reconcile(self, driver, r):
        fn = getattr(driver, "reconcile_write", None)
        if not callable(fn):
            return None
        try:
            rr = fn("submit", {}, r)
            return rr.get("confirmed") if isinstance(rr, dict) else None
        except Exception:
            return None

    def _outcome_from(self, r, commit=False):
        if isinstance(r, Outcome):
            return r
        if not isinstance(r, dict):
            return Outcome(status=RESULT_OK, result=r)
        if r.get("error"):
            return Outcome(status=RESULT_FAILED, result=r, reason=str(r.get("error"))[:200])
        created = None
        for k in ("created_id", "confirmationId", "id"):
            if r.get(k):
                created = r.get(k)
                break
        return Outcome(status=RESULT_OK, result=r.get("observation", r),
                       created_id=created, raw=r)

    # -- authoritative read-back ------------------------------------------------------------------
    def read_state(self, paths):
        """Read the persisted portal state. Returns BOTH the EMR and the two payer portals so a
        workflow's read-back can verify a landed submission/appeal/documentation. Reads the environment's
        own localStorage only (oracle-blind)."""
        driver = self.driver
        # 1) preferred: the (to-be-added) GuiEnvReal.read_recovery_state() extension.
        fn = getattr(driver, "read_recovery_state", None)
        if callable(fn):
            try:
                st = fn()
                if isinstance(st, dict):
                    return st
            except Exception:
                pass
        # 2) self-serve: read localStorage 'portals_state' directly (no environments.py edit required).
        page = getattr(driver, "page", None)
        if page is not None:
            try:
                raw = page.evaluate(
                    "() => { try { return localStorage.getItem('portals_state'); }"
                    " catch(e){ return null; } }")
                data = json.loads(raw) if raw else {}
                if not isinstance(data, dict):
                    data = {}
                return {
                    "full_state": data.get("emr") if isinstance(data.get("emr"), dict) else {},
                    "payer_a_state": data.get("payerA") if isinstance(data.get("payerA"), dict) else {},
                    "payer_b_state": data.get("payerB") if isinstance(data.get("payerB"), dict) else {},
                    "fax": data.get("fax") if isinstance(data.get("fax"), dict) else {},
                }
            except Exception:
                pass
        # 3) degraded fallback: only the EMR the driver already cached.
        fs = getattr(driver, "full_state", None)
        return {"full_state": fs if isinstance(fs, dict) else {},
                "payer_a_state": {}, "payer_b_state": {}}

    # -- result classification --------------------------------------------------------------------
    def classify_result(self, result):
        """Map a primitive result/Outcome to the outcome vocabulary."""
        if isinstance(result, Outcome):
            st = result.status
            if st in (RESULT_OK, RESULT_UNKNOWN, RESULT_FAILED, RESULT_ALREADY_REALIZED):
                return st
            return RESULT_OK
        if isinstance(result, dict):
            if result.get("error"):
                return RESULT_FAILED
            if result.get("state_changed") is False:
                return RESULT_UNKNOWN
            return RESULT_OK
        return RESULT_OK
