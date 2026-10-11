"""Unified GUI control model: ONE accessible-control parser shared by GuiEnvReal.list_controls
and the harness action service's list_controls view (runner/harness/action_service.py), so the two never drift. ONE accessible-name implementation
(accName) is shared by BOTH the list parser (CONTROL_JS) and the per-ref readback (READBACK_JS), so a control
carries the SAME semantic label everywhere (the write receipt reads the label it lists). Benchmark-neutral; no
gold.
"""

# ONE accessible-name implementation, injected into every JS parser so labels never drift.
_ACC_NAME_JS = r"""
  function accName(el){
    if(!el) return '';
    const aria=el.getAttribute('aria-label'); if(aria&&aria.trim()) return aria.trim();
    const lb=el.getAttribute('aria-labelledby');
    if(lb){ const t=lb.split(/\s+/).map(id=>{const n=document.getElementById(id); return n?(n.innerText||''):'';}).join(' ').trim(); if(t) return t; }
    try{ if(el.labels&&el.labels.length){ const t=Array.from(el.labels).map(x=>x.innerText||'').join(' ').trim(); if(t) return t; } }catch(e){}
    const wl=el.closest('label'); if(wl&&wl.innerText&&wl.innerText.trim()) return wl.innerText.trim();
    return (el.getAttribute('placeholder')||el.getAttribute('name')||el.getAttribute('title')||'').trim();
  }
  function sectionName(el){
    if(!el||!el.closest) return '';
    const sec=el.closest('form,fieldset,section,[role=region],[role=dialog],[data-section],[data-testid*=section]');
    if(!sec) return '';
    let nm=sec.getAttribute('aria-label')||'';
    if(!nm){ try{ const h=sec.querySelector('legend,h1,h2,h3,h4,[role=heading]'); if(h) nm=(h.innerText||'').trim(); }catch(e){} }
    if(!nm) nm=sec.getAttribute('data-section')||sec.getAttribute('id')||sec.getAttribute('data-testid')||'';
    return String(nm||'').slice(0,60);
  }
"""

CONTROL_JS = "() => {\n" + _ACC_NAME_JS + r"""
  const SEL='button,a,[role=button],input,select,textarea,[role=combobox],[role=checkbox],[role=radio],[role=switch]';
  const els=Array.from(document.querySelectorAll(SEL));
  return els.map((el,i)=>{
    let ref=el.getAttribute('data-mh-ref'); if(ref===null){ ref=el.getAttribute('data-testid')||String(i); }
    const tag=el.tagName.toLowerCase();
    const type=(el.getAttribute('type')||el.getAttribute('role')||'').toLowerCase();
    const nm=accName(el);
    const inner=(el.innerText||'').trim();
    const label=String(nm||inner||'').slice(0,80);
    const name=el.getAttribute('name')||el.getAttribute('id')||'';
    const required=!!(el.required||el.getAttribute('aria-required')==='true');
    const value=String(el.value||'').slice(0,120);
    const disabled=!!(el.disabled||el.getAttribute('aria-disabled')==='true');
    let visible=true;
    try{ const r=el.getBoundingClientRect(); visible=!!(r.width>0&&r.height>0)&&el.getAttribute('aria-hidden')!=='true'; }catch(e){}
    let role=tag==='select'?'select':(tag==='textarea'?'textbox':(tag==='input'?(type||'input'):(tag==='button'?'button':(el.getAttribute('role')||tag))));
    let checked=null, selIdx=null, selText=null, placeholderSel=null;
    if(type==='checkbox'||type==='radio'||role==='checkbox'||role==='radio'||role==='switch'){ checked=!!el.checked; role=(type||role); }
    if(tag==='select'){ try{ selIdx=el.selectedIndex; const o=el.options[el.selectedIndex]; selText=o?String(o.text||o.value||''):''; placeholderSel=(selIdx<=0)||(String((o&&o.value)||'')===''); }catch(e){} }
    const ariaSel=el.getAttribute('aria-selected');
    const commit=(type==='submit')||(tag==='button'&&/submit|send|transmit|finaliz|\bfile\b/i.test(label));
    return {ref, role, label, section:sectionName(el), name, id:el.getAttribute('id')||'', testid:el.getAttribute('data-testid')||'',
            readonly:((el.readOnly===true)||(el.getAttribute('aria-readonly')==='true')),
            required, value, disabled, visible, checked, selected_index:selIdx, selected_text:selText,
            placeholder_selected:placeholderSel, aria_selected:ariaSel,
            options: tag==='select'?Array.from(el.options).map(o=>o.value||o.text):[], commit:!!commit};
  }).filter(c=>c.label||c.role==='select'||c.name);
}
"""

# Read ONE control back by ref after a write, with the SAME accessible label.
READBACK_JS = "(r) => {\n" + _ACC_NAME_JS + r"""
  const el=document.querySelector('[data-mh-ref="'+r+'"]')||document.querySelector('[data-testid="'+r+'"]');
  if(!el) return {found:false};
  let sel=null;
  if(el.tagName==='SELECT'){ try{const o=el.options[el.selectedIndex]; sel=o?String(o.text||o.value||''):'';}catch(e){} }
  let val=String(el.value||'');
  // CUSTOM dropdown (button / combobox trigger, not a native input/select): it has no form .value -- the
  // SELECTED value is the text DISPLAYED on the trigger. Read that (minus arrow glyphs) so a landed custom
  // selection is detectable instead of always reading '' (which false-flagged write_not_landed).
  const tag=el.tagName;
  const custom=(tag==='BUTTON'||el.getAttribute('role')==='combobox'||el.getAttribute('role')==='button'
                ||(tag!=='INPUT'&&tag!=='TEXTAREA'&&tag!=='SELECT'));
  if(!val && custom){
    let t=String(el.innerText||el.textContent||'').replace(/[\u25b2\u25b4\u25be\u25bc\u2303\u2304\u02c5\u02c4]/g,'');
    val=t.replace(/\s+/g,' ').trim();
  }
  return {found:true, value:val, checked:(el.checked===undefined?null:!!el.checked),
          selected_text:sel, label:accName(el), section:sectionName(el)};
}
"""


def norm_value(v):
    import re
    return re.sub(r"\s+", " ", str(v if v is not None else "")).strip().lower()


def is_control_complete(c):
    """Role-aware 'this control already holds a value' test."""
    if not isinstance(c, dict):
        return True
    if not c.get("visible", True) or c.get("disabled"):
        return True
    role = str(c.get("role") or "").lower()
    if role in ("checkbox", "radio", "switch"):
        return bool(c.get("checked"))
    if role == "select":
        si = c.get("selected_index")
        return (si is not None and si > 0) and not c.get("placeholder_selected")
    if role in ("button", "a", "link", "submit"):
        return True
    return bool(str(c.get("value") or "").strip())


def write_landed(attempted, receipt):
    """Did the agent's attempted value land in the read-back control? Oracle-blind; never gold."""
    if not receipt or not receipt.get("found"):
        return False
    observed = receipt.get("selected_text") or receipt.get("value") or ""
    att = norm_value(attempted)
    if not att:
        return receipt.get("checked") is True
    obs = norm_value(observed)
    return bool(obs) and (att in obs or obs in att)


# ---- Field ontology (generic, no gold). Read by GUI recovery's field binding
# (workspaces/hab/h0/recovery/gui_completion.py). The write-verify middleware keeps its own copy, with the misroute
# reconciler (workspaces/hab/h0/middleware/write_verify.py). ----
# GENERIC UI field aliases (canonical concept -> public synonyms). NOT dataset heuristics: never "value looks
# like AET\\d+ -> Member ID". Only accessible-name semantics of controls / agent-declared field intents.
FIELD_ALIASES = {
    "member_id": ["member id", "member number", "member no", "subscriber id", "subscriber number",
                  "insurance id", "policy number", "member"],
    "date_of_birth": ["date of birth", "dob", "birth date", "birthdate"],
    "patient_search": ["patient search", "search patient", "find patient", "member search", "lookup patient"],
    "request_type": ["request type", "authorization type", "service type", "auth type", "review type"],
    "diagnosis": ["diagnosis", "diagnosis code", "icd", "dx code"],
    "cpt": ["cpt", "procedure code", "hcpcs", "service code", "cpt/hcpcs"],
    "provider": ["provider", "npi", "servicing provider", "requesting provider", "rendering provider"],
    "clinical_justification": ["clinical justification", "medical necessity", "clinical indication",
                               "justification", "clinical note", "clinical rationale"],
    "fax_number": ["fax number", "fax", "recipient fax"],
    "patient_name": ["patient name", "member name", "last name", "first name"],
}


def _field_concept(text):
    """Map a control accessible-name / agent field intent to a canonical concept, or None if unknown. Longest
    matching synonym wins so 'member search' -> patient_search (not member_id)."""
    n = norm_value(text)
    if not n:
        return None
    best, best_len = None, 0
    for concept, syns in FIELD_ALIASES.items():
        for syn in syns:
            if syn in n and len(syn) > best_len:
                best, best_len = concept, len(syn)
    return best


import re as _re
_LOOKUP_SECTION_RE = _re.compile(r"\b(search|lookup|look ?up|find (?:a )?(?:patient|member)|eligibility|member search|patient search)\b", _re.I)


def _is_lookup_section(section):
    """A form section that is a LOOKUP/search surface (patient/member search, eligibility) -- distinct from
    the payload form where the checkpoint reads the value. Public DOM section semantics; no gold."""
    return bool(_LOOKUP_SECTION_RE.search(str(section or "")))


# --- ONE shared terminal-COMMIT classifier (irreversible submit/send) used by every gate, so a
# non-standard button ref (fax-send-button / primary-action / confirm-button / button-17) is still caught via
# the tool + the control's commit flag / label, not each gate grepping the ref string. ---
_TERMINAL_REF_RE = _re.compile(r"(send[-_ ]?fax|sendfax|send[-_ ]?appeal|transmit|submit|file[-_ ]?appeal|finali[sz])", _re.I)
_TERMINAL_LABEL_RE = _re.compile(r"\b(submit|send|transmit|finali[sz]|file\s+(?:claim|appeal|authorization))\b", _re.I)
# A NAVIGATION-submit ('Submit to Insurance/Payer/Portal') is a hop INTO the payer system, not
# the terminal prior-auth submit. CONTROL_JS marks it commit=true (label contains 'submit'), so the runner
# gates (precommit / attachment / collection) would otherwise block this navigation as if it were the final
# commit -- BEFORE the agent even reaches the payer portal. Exempt it here so every gate that shares this
# classifier treats it as navigation, mirroring _NAV_SUBMIT_RE in the recovery workflow.
_NAV_SUBMIT_RE = _re.compile(r"submit\s+to\s+(?:the\s+)?(?:payer|insurance|portal|plan|carrier|health\s*plan)", _re.I)


def is_terminal_action(tool, ref="", controls=None):
    """True if the action is a terminal COMMIT. tool=='submit' -> yes; else resolve the control by ref from
    `controls` and use its commit flag / submit-like label; finally fall back to a terminal-looking ref. Robust
    to a non-standard ref because it prefers the commit flag + label over the ref substring."""
    if str(tool or "").lower() == "submit":
        return True
    ref = str(ref or "")
    for c in (controls or []):
        if str(c.get("ref")) == ref:
            if _NAV_SUBMIT_RE.search(str(c.get("label") or "")):
                return False                              # navigation to the payer, not a commit
            if c.get("commit") is True:
                return True
            if _TERMINAL_LABEL_RE.search(str(c.get("label") or "")):
                return True
            break
    if _NAV_SUBMIT_RE.search(_re.sub(r"[-_]+", " ", ref)):
        return False                                  # nav-submit by ref (hyphen/underscore form)
    return bool(_TERMINAL_REF_RE.search(ref))
