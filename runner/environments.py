"""Environment adapters — dispatch by unified task `environment.type`.

base: reset() / available_tools() / call_tool(name,args) / teardown().
FhirEnv is real (hits the HAPI FHIR server). GuiEnv / ToolSandboxEnv are skeleton stubs
(to be filled with Playwright / agentlego). Registry maps environment.type -> adapter.
"""
import os, json, urllib.request, urllib.parse, urllib.error

class EnvironmentAdapter:
    type = "base"
    def __init__(self, task, **kw): self.task = task; self.cfg = (task.get("environment") or {}).get("config", {})
    def reset(self): ...
    def available_tools(self): return [t["name"] for t in self.task.get("available_tools", [])]
    def call_tool(self, name, args): raise NotImplementedError
    def teardown(self): ...

class FhirEnv(EnvironmentAdapter):
    type = "fhir"
    def __init__(self, task, fhir_base=None, workspace=None, aug_dir=None, **kw):
        super().__init__(task)
        self.base = fhir_base or os.environ.get("FHIR_BASE_URL") or self.cfg.get("default") or "http://localhost:38080/fhir"
        self.workspace = workspace or "/tmp/mh_workspace"
        self.aug_dir = aug_dir
        os.makedirs(self.workspace, exist_ok=True)
        self._lab = None
    def _get(self, path):
        req = urllib.request.Request(self.base + path, headers={"Accept": "application/fhir+json"})
        try:
            return json.load(urllib.request.urlopen(req, timeout=30))
        except urllib.error.HTTPError as ex:
            return {"error": "HTTP %s" % ex.code, "detail": ex.read().decode("utf-8", "ignore")[:300], "query": path}
        except Exception as ex:
            return {"error": repr(ex), "query": path}
    def _normalize_search(self, rt, params):
        """Make FHIR search forgiving so any agent`s intuitive `patient=<MRN>` works.
        Patient has NO patient/subject search param -> route to `identifier`; clinical resources take
        an MRN via chained `patient.identifier` (a bare MRN is not a valid reference target)."""
        p = dict(params)
        PATIENTISH = ("patient", "subject", "mrn", "patient_id", "patientId", "patientID")
        if rt == "Patient":
            for k in PATIENTISH:
                if k in p:
                    p.setdefault("identifier", p.pop(k))
        else:
            for k in ("patient", "subject"):
                if k in p:
                    v = str(p[k])
                    if not v.startswith(("Patient/", "urn:", "http")):  # bare MRN -> chained identifier search
                        p["%s.identifier" % k] = p.pop(k)
            for k in ("mrn", "patient_id", "patientId", "patientID"):
                if k in p:
                    p.setdefault("patient.identifier", p.pop(k))
        return p

    def call_tool(self, name, args):
        if name in _FHIR_GRANULAR:
            rt, cat = _FHIR_GRANULAR[name]; a = dict(args or {}); a["resourceType"] = rt
            if cat and "category" not in a: a["category"] = cat
            return self.call_tool("fhir_search", a)
        if name in _FHIR_GRANULAR_CREATE:
            res = dict((args or {}).get("resource", args or {})); res["resourceType"] = _FHIR_GRANULAR_CREATE[name]
            return self.call_tool("fhir_create", {"resource": res})
        if name == "fhir_search":
            rt = args.get("resourceType", ""); params = {k: v for k, v in args.items() if k != "resourceType"}
            params = self._normalize_search(rt, params)
            return self._get(f"/{rt}?" + urllib.parse.urlencode(params))
        if name == "fhir_read":
            return self._get(f"/{args['resourceType']}/{args['id']}")
        if name == "fhir_create":
            res = args.get("resource", args); rt = res.get("resourceType", "")
            data = json.dumps(res).encode()
            req = urllib.request.Request(f"{self.base}/{rt}", data=data, method="POST",
                                         headers={"Content-Type": "application/fhir+json"})
            try:
                return json.load(urllib.request.urlopen(req, timeout=30))
            except urllib.error.HTTPError as ex:
                return {"error": "HTTP %s" % ex.code, "detail": ex.read().decode("utf-8", "ignore")[:300]}
            except Exception as ex:
                return {"error": repr(ex)}
        if name == "get_lab_reference_range":
            if self._lab is None:
                import importlib.util, sys
                p = os.path.join(os.path.dirname(__file__), "..", "benchmark_dataprocess", "PhysicianBench", "lab_ref.py")
                spec = importlib.util.spec_from_file_location("lab_ref", p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); self._lab = m
            return self._lab.get_lab_reference_range(args.get("loinc"), args.get("sex"), args.get("age"), args.get("unit"))
        if name == "write_file":
            # safe join: keep subdirs (upstream verifiers may expect them) but block path traversal
            rel = os.path.normpath(args.get("path", "out.txt")).lstrip("/")
            if rel.startswith(".."):
                return {"error": "path traversal rejected"}
            # workspace already ends with .../workspace/output — strip that prefix if the agent
            # echoed the conventional path (else we'd double-nest and native_pytest wouldn't find it)
            for pre in ("workspace/output/", "workspace/", "output/"):  # self.workspace already IS .../workspace/output -> also strip bare output/ (else output/X nests to output/output/X)
                if rel.startswith(pre):
                    rel = rel[len(pre):]
                    break
            fp = os.path.join(self.workspace, rel)
            os.makedirs(os.path.dirname(fp) or self.workspace, exist_ok=True)
            open(fp, "w").write(args.get("content", "")); return {"written": fp}
        return {"error": f"unknown tool {name}"}

class GuiEnvMock(EnvironmentAdapter):
    """HealthAdminBench v0: MOCK portal — browser actions mutate an in-memory full_state, which the
    JMESPath checkpoints score against. Proves the GUI substrate path without Playwright/npm/internet.
    v1 = drive the real NextJS portal via Playwright (harness/real_obs.py) and capture full_state.
    """
    type = "gui"
    def __init__(self, task, **kw):
        super().__init__(task); self.full_state = None
    def reset(self):
        self.full_state = {"agentActions": {}, "signals": {}, "triageNotes": "", "fields": {}, "_page": None}
    def call_tool(self, name, args):
        fs = self.full_state
        if name == "navigate":
            fs["_page"] = args.get("url") or args.get("target")
        elif name == "click":
            t = args.get("target", ""); fs["agentActions"][t] = True
        elif name == "type":
            f = args.get("field") or args.get("target", ""); txt = args.get("text", "")
            fs["fields"][f] = txt
            if "note" in f.lower(): fs["triageNotes"] = txt
        elif name == "select":
            fs["agentActions"][args.get("field") or args.get("target", "")] = args.get("value")
        elif name in ("upload", "submit"):
            fs["agentActions"][f"{name}_{args.get('target','')}".strip('_')] = True
        elif name == "snapshot":
            return {"full_state": fs}
        else:
            return {"error": f"unknown gui tool {name}"}
        return {"ok": True, "state_keys": list(fs["agentActions"].keys())}

class ToolSandboxEnv(EnvironmentAdapter):
    """MedCTA environment. mode=real (v1, DEFAULT): the 5 tools execute for real
    (ImageDescription / RegionAttributeDescription / OCR via local VLM backend; Calculator; GoogleSearch
    frozen corpus). mode=replay (v0): returns cached tool outputs from the reference trace so a
    ReplayAgent can validate ToolAcc/ArgAcc/Gacc WITHOUT a GPU. Select via MH_TOOL_MODE (real|replay).
    The image is resolved from task.context.images[0].path under MH_MEDCTA_IMG_ROOT.
    """
    type = "tool_sandbox"
    def __init__(self, task, **kw):
        super().__init__(task)
        self.mode = os.environ.get("MH_TOOL_MODE", "real")
        trace = (task.get("reference") or {}).get("reference_trace") or []
        self._cached = [ev.get("content") for ev in trace if ev.get("role") == "tool"]
        self._i = 0
        self.image_path = self._resolve_image(task)
    def _resolve_image(self, task):
        imgs = (task.get("context") or {}).get("images") or []
        if not imgs: return None
        rel = imgs[0].get("path") or ""
        root = os.environ.get("MH_MEDCTA_IMG_ROOT", os.path.join(
            os.path.dirname(__file__), "..", "benchmark", "MedCTA", "opencompass", "data", "medcta_dataset"))
        fp = os.path.join(root, rel)
        return fp if os.path.exists(fp) else rel
    def reset(self):
        self._i = 0
    def call_tool(self, name, args):
        if self.mode == "replay":
            out = self._cached[self._i] if self._i < len(self._cached) else {"replayed": True}
            self._i += 1
            return {"tool": name, "args": args, "output": out, "mode": "replay"}
        import tools_medcta
        out = tools_medcta.run_tool(name, args or {}, self.image_path)
        return {"tool": name, "args": args, "output": out, "mode": "real"}


class GuiEnvReal(EnvironmentAdapter):
    """HealthAdminBench v1: drives the REAL Next.js portal via Playwright. Gives the agent a readable
    OBSERVATION each step: visible page text + an enumerated list of interactive elements tagged with
    data-mh-ref indices (self-contained, no browsergym/loguru). The agent addresses elements by `ref`
    (click/type/select accept {"ref": N}); a raw CSS selector or visible text still works as fallback.
    full_state is read from the portal's localStorage key portals_state -> .emr (what the JMESPath
    checkpoints score). Requires the portal at MH_PORTAL_BASE (default http://localhost:3002) and
    chromium. v0 mock = GuiEnvMock (MH_GUI_MODE=mock)."""
    type = "gui"

    _OBS_JS = r"""() => {
      document.querySelectorAll('[data-mh-ref]').forEach(e => e.removeAttribute('data-mh-ref'));
      const SEL = 'a,button,input,select,textarea,[role=button],[role=link],[role=tab],[role=menuitem],[role=checkbox],[role=radio],[role=switch],[onclick],[tabindex]';
      const els = Array.from(document.querySelectorAll(SEL));
      const out = []; let i = 0;
      for (const el of els) {
        const r = el.getBoundingClientRect();
        if (!(r.width > 0 && r.height > 0 && el.offsetParent !== null)) continue;
        el.setAttribute('data-mh-ref', String(i));
        let label = (el.innerText || '').trim();
        if (!label) label = el.getAttribute('aria-label') || el.getAttribute('placeholder') || el.value || el.getAttribute('name') || el.getAttribute('title') || '';
        label = String(label).replace(/\s+/g, ' ').trim().slice(0, 80);
        out.push({ref: i, tag: el.tagName.toLowerCase(), type: (el.getAttribute('type') || el.getAttribute('role') || ''), label: label, value: String(el.value || '').slice(0, 40)});
        i++;
      }
      let text = '';
      try { text = (document.body.innerText || '').replace(/\n{3,}/g, '\n\n').trim().slice(0, 1600); } catch (e) {}
      return {elements: out, text: text};
    }"""

    def __init__(self, task, **kw):
        super().__init__(task)
        import urllib.parse as up
        self.base = os.environ.get("MH_PORTAL_BASE", "http://localhost:3002")
        site = (self.cfg.get("website") or {})
        path = up.urlparse(site.get("url") or "").path or "/"
        self.start_url = self.base + path
        self._pw = self._browser = self.page = None
        self.full_state = None
        self._init_obs = None

    def reset(self):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True, args=["--no-sandbox"])
        self.page = self._browser.new_page()
        self.page.goto(self.start_url, wait_until="domcontentloaded", timeout=45000)
        self.page.wait_for_timeout(800)
        self._read_state()
        self._init_obs = self._snap()

    def initial_observation(self):
        return self._init_obs

    def _read_state(self):
        try:
            raw = self.page.evaluate("() => { try { return localStorage.getItem('portals_state'); } catch(e){ return null; } }")
            st = json.loads(raw) if raw else None
            self.full_state = (st.get("emr") if isinstance(st, dict) else None) or {}
        except Exception:
            pass
        return self.full_state

    def _observe(self):
        try:
            o = self.page.evaluate(self._OBS_JS)
        except Exception as e:
            return "(observation failed: %r)" % e
        lines = ["URL: %s" % self.page.url, "--- PAGE TEXT ---", o.get("text", "")[:1600],
                 "--- INTERACTIVE ELEMENTS (use the ref number with click/type/select) ---"]
        for el in o.get("elements", []):
            t = ("[%s]" % el["type"]) if el.get("type") else ""
            v = (" value=%r" % el["value"]) if el.get("value") else ""
            lines.append("[ref=%d] %s%s %r%s" % (el["ref"], el["tag"], t, el.get("label", ""), v))
        return "\n".join(lines)

    def _snap(self):
        self._read_state()
        return {"ok": True, "url": self.page.url, "title": self.page.title(), "observation": self._observe()}

    def _sel(self, a):
        if a.get("ref") not in (None, ""):
            return ('[data-mh-ref="%s"]' % a.get("ref")), "css"
        s = a.get("selector") or a.get("target") or a.get("field") or ""
        if s.startswith(("#", ".", "[")) or (s.split(":")[0] in ("button", "a", "div", "span", "input", "select", "textarea")):
            return s, "css"
        return s, "text"

    def _click(self, a):
        sel, kind = self._sel(a)
        if kind == "css":
            self.page.click(sel, timeout=8000)
        else:
            self.page.get_by_text(sel, exact=False).first.click(timeout=8000)

    def _fill(self, a, text):
        sel, kind = self._sel(a)
        if kind == "css":
            self.page.fill(sel, text, timeout=8000)
        else:
            self.page.get_by_label(sel).first.fill(text, timeout=8000)

    def call_tool(self, name, args):
        a = args or {}
        try:
            if name == "navigate":
                tgt = a.get("url") or a.get("target") or "/"
                import urllib.parse as _up
                pth = _up.urlparse(tgt).path if "://" in tgt else (tgt if tgt.startswith("/") else "/" + tgt)
                self.page.goto(self.base + (pth or "/"), wait_until="domcontentloaded", timeout=45000)
            elif name == "click":
                self._click(a)
            elif name == "type":
                self._fill(a, a.get("text", ""))
            elif name == "select":
                sel, _k = self._sel(a)
                self.page.select_option(sel, a.get("value"))
            elif name == "submit":
                if a.get("ref") or a.get("selector") or a.get("target"):
                    self._click(a)
                else:
                    self.page.click("button[type=submit]", timeout=8000)
            elif name == "snapshot":
                return self._snap()
            else:
                return {"error": "unknown gui tool %s" % name}
        except Exception as e:
            try:
                self.page.wait_for_timeout(300)
            except Exception:
                pass
            r = self._snap()
            r["error"] = repr(e)
            return r
        self.page.wait_for_timeout(500)
        return self._snap()

    def teardown(self):
        try: self._read_state()
        except Exception: pass
        try: self._browser.close()
        except Exception: pass
        try: self._pw.stop()
        except Exception: pass

_FHIR_GRANULAR = {
    "fhir_patient_search_demographics": ("Patient", None),
    "fhir_condition_search_problems": ("Condition", None),
    "fhir_observation_search_labs": ("Observation", "laboratory"),
    "fhir_observation_search_vitals": ("Observation", "vital-signs"),
    "fhir_observation_search_social_history": ("Observation", "social-history"),
    "fhir_medication_request_search_orders": ("MedicationRequest", None),
    "fhir_procedure_search_orders": ("Procedure", None),
    "fhir_document_reference_search_clinical_notes": ("DocumentReference", None),
    "fhir_service_request_search": ("ServiceRequest", None),
}
_FHIR_GRANULAR_CREATE = {
    "fhir_medication_request_create": "MedicationRequest",
    "fhir_service_request_create": "ServiceRequest",
    "fhir_communication_create_message": "Communication",
    "fhir_appointment_create": "Appointment",
}
# Upstream PhysicianBench granular tool surface (replaces generic fhir_search for the agent prompt).
FHIR_GRANULAR_TOOLS = [
    {"name": "fhir_patient_search_demographics", "signature": "(patient) -> demographics"},
    {"name": "fhir_condition_search_problems", "signature": "(patient) -> problems/diagnoses"},
    {"name": "fhir_observation_search_labs", "signature": "(patient) -> laboratory results"},
    {"name": "fhir_observation_search_vitals", "signature": "(patient) -> vital signs"},
    {"name": "fhir_observation_search_social_history", "signature": "(patient) -> social history"},
    {"name": "fhir_medication_request_search_orders", "signature": "(patient) -> medication orders"},
    {"name": "fhir_procedure_search_orders", "signature": "(patient) -> procedures"},
    {"name": "fhir_document_reference_search_clinical_notes", "signature": "(patient) -> clinical notes"},
    {"name": "fhir_service_request_search", "signature": "(patient) -> service requests/orders"},
    {"name": "fhir_read", "signature": "(resourceType, id) -> read one resource by id"},
    {"name": "fhir_medication_request_create", "signature": "(resource) -> place a medication order"},
    {"name": "fhir_service_request_create", "signature": "(resource) -> place a lab/imaging/service order"},
    {"name": "fhir_communication_create_message", "signature": "(resource) -> send a message"},
    {"name": "fhir_appointment_create", "signature": "(resource) -> schedule an appointment"},
    {"name": "get_lab_reference_range", "signature": "(loinc, sex, age, unit) -> reference range"},
    {"name": "write_file", "signature": "(path, content) -> save the required deliverable"},
]

ENV_REGISTRY = {"fhir": FhirEnv, "gui": GuiEnvMock, "tool_sandbox": ToolSandboxEnv}

def make_env(task, **kw):
    et = (task.get("environment") or {}).get("type")
    if et == "gui":
        # SAFE DEFAULT = mock. The real Playwright portal needs a launchable chromium, which the
        # LOGIN node lacks (playwright imports fine there but browser launch crashes — so an
        # import probe is unreliable). Default-real therefore made run.py / run_batch crash on
        # every GUI task at login. Require an explicit opt-in: set MH_GUI_MODE=real on a GPU node
        # (the HAB sbatch does). Mock is qualified as mock_inmemory in provenance, so the default
        # downgrade stays honest.
        cls = GuiEnvReal if os.environ.get("MH_GUI_MODE") == "real" else GuiEnvMock
        return cls(task, **kw)
    return ENV_REGISTRY[et](task, **kw)
