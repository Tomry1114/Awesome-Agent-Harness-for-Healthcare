"""Advisory-only VERIFY check: ORDER SELF-CONSISTENCY.

Checks the internal consistency of the agent's OWN ServiceRequest orders -- it NEVER authors a
clinical decision, NEVER supplies a code/value, NEVER reads the checkpoint graders. It only asks:
"does the machine-readable order the agent wrote agree with what the agent itself said it was ordering?"

Three defect classes (each attributable, oracle-blind, content-free):
  1. DUP_CODE           -- one code_code stamped on >=2 different tests (a lab keying on the code runs the wrong test)
  2. INTENT_CODE_MISMATCH-- the order's own free-text intent names analyte A, but its coded display is analyte B
  3. SUPERSEDE_NOT_REVOKED- the agent declares an order erroneous/replaced but leaves the bad order active

Advisory output only: a list of findings the harness would surface as a nudge ("your order X's code says
albumin but you wrote 'dexamethasone level' -- reconcile"). It does not tell the agent WHICH code to use.
"""
import re

# known orderable lab analytes / imaging modalities (tokens, lowercase). Used to compare the order's
# free-text INTENT against its coded DISPLAY. Not a value source -- only a "same analyte or not" test.
_ANALYTES = {
    "dexamethasone", "cortisol", "albumin", "potassium", "sodium", "chloride", "creatinine",
    "glucose", "calcium", "metanephrine", "metanephrines", "aldosterone", "renin",
    "dehydroepiandrosterone", "dhea", "acth", "catecholamine", "catecholamines", "cortisone",
}
# imaging tokens are treated as their own analyte-family
_IMAGING = {"ct", "mri", "ultrasound", "pet"}
_SUPERSEDE = re.compile(r"errone|replaces?\s+(order|service|sr)|corrected order|entered.?in.?error|supersed|instead of order", re.I)
_TESTNOUN = r"(?:level|test|assay|measurement|study|screen|value|labs?|draw|order|suppression)"
_PANEL = re.compile(r"\bpanel\b|\bBMP\b|\bCMP\b|\bprofile\b", re.I)


def _norm(a):
    return "dhea" if a in ("dehydroepiandrosterone", "dhea") else ("metanephrine" if a.startswith("metanephrine") else a)


def _analytes_in(text):
    """Every analyte token mentioned (used for the CODED DISPLAY side)."""
    t = (text or "").lower()
    return {_norm(a) for a in _ANALYTES if re.search(r"\b" + re.escape(a) + r"\b", t)}


def _named_test_analytes(text):
    """Only analytes the free-text names AS THE ORDERED TEST (analyte adjacent to a test-noun, or leading),
    NOT ones merely mentioned as clinical context. Kills false positives from contextual mentions."""
    t = (text or "").lower().strip()
    found = set()
    for a in _ANALYTES:
        if (re.search(r"\b" + re.escape(a) + r"\b\s+\w{0,6}?\s?" + _TESTNOUN, t)
                or re.search(_TESTNOUN + r"\s+(?:of\s+)?\b" + re.escape(a) + r"\b", t)
                or re.match(r"\b" + re.escape(a) + r"\b", t)):
            found.add(_norm(a))
    return found


def _order_view(args):
    """Split an order's args into (coded display text) and (agent free-text intent), content-free."""
    code = args.get("code_code")
    display = args.get("code_display") or ""
    intent_parts = []
    for k, v in args.items():
        if k in ("code_code", "code_display", "code_system", "category_code", "category_display",
                 "category_system") or not isinstance(v, str):
            continue
        intent_parts.append(v)
    return code, display, " ".join(intent_parts)


def scan_orders(orders):
    """orders: list of ServiceRequest arg-dicts (in creation order). Returns advisory findings."""
    findings = []
    by_code = {}
    for i, a in enumerate(orders):
        code, display, intent = _order_view(a)
        by_code.setdefault(code, []).append((i, display))
        # 2. INTENT_CODE_MISMATCH: fire ONLY when the coded display's analyte is NEVER mentioned in the order's
        # own free-text intent (the agent doesn't acknowledge what it actually coded) AND the intent NAMES a
        # different analyte as the ordered test. Panels are exempt (they legitimately carry many analytes).
        ad = _analytes_in(display)
        intent_all = _analytes_in(intent)
        named = _named_test_analytes(intent) - ad
        if ad and named and not (ad & intent_all) and not _PANEL.search(display):
            findings.append({
                "class": "INTENT_CODE_MISMATCH", "order_index": i, "code": code,
                "coded_analyte": sorted(ad), "intent_analyte": sorted(named),
                "detail": "order names a %s test in its own text but the coded display is %s (never mentioned) "
                          "-- the lab will run %s" % (sorted(named), sorted(ad), sorted(ad))})
        # 3. SUPERSEDE_NOT_REVOKED: agent declares this/another order erroneous but statuses stay active
        if _SUPERSEDE.search(intent) or _SUPERSEDE.search(display):
            active_dupes = [(j, d) for (j, d) in by_code.get(code, []) if j != i]
            if any(orders[j].get("status", "active") == "active" for j, _ in active_dupes) or \
               all(o.get("status", "active") == "active" for o in orders):
                findings.append({
                    "class": "SUPERSEDE_NOT_REVOKED", "order_index": i, "code": code,
                    "detail": "order text flags a prior order erroneous/replaced, but no order was set "
                              "revoked/entered-in-error (all remain active)"})
    # 1. DUP_CODE: one code_code on >=2 distinct displays
    for code, lst in by_code.items():
        displays = {d.strip().lower() for _, d in lst if d}
        if len(displays) > 1:
            findings.append({
                "class": "DUP_CODE", "code": code, "order_indices": [i for i, _ in lst],
                "displays": sorted({d for _, d in lst if d}),
                "detail": "code %s is stamped on %d DIFFERENT tests %s -- a system keying on the code runs the wrong one"
                          % (code, len(displays), sorted(displays))})
    return findings
