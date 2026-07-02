"""Environment adapters — dispatch by unified task `environment.type`.

base: reset() / available_tools() / call_tool(name,args) / teardown().
FhirEnv is real (hits the HAPI FHIR server). GuiEnv / ToolSandboxEnv are skeleton stubs
(to be filled with Playwright / agentlego). Registry maps environment.type -> adapter.
"""
import os, json, urllib.request, urllib.parse, urllib.error

class EnvironmentAdapter:
    """BENCHMARK ADAPTER CONTRACT (Codex B). Every benchmark registers ONE env class in ENV_REGISTRY
    and the runner drives it through this fixed surface — adding a benchmark = register a class, not
    edit the loop's execution path:
        reset()                      -> initialize episode (load_task handled by load_task() upstream)
        available_tools()            -> list[str] tools offered to the agent this task
        call_tool(name, args)        -> CanonicalResult-shaped dict (execute one action)
        capabilities()               -> four-state manifest {implemented/available/authorized/healthy}
        teardown()                   -> release resources
        initial_observation()        -> optional first CanonicalObservation (envs that have one)
    Source-metric evaluation (evaluate_source_metric) and artifact collection currently live in the
    scoring layer (scoring.run_checkpoint dispatch); the contract spans env+scorer and full unification
    of evaluate_source_metric into the adapter is the staged next step. Conformance is enforced by
    test_conformance.test_benchmark_adapter_contract over every ENV_REGISTRY class."""
    type = "base"
    def __init__(self, task, **kw): self.task = task; self.cfg = (task.get("environment") or {}).get("config", {})
    def reset(self): ...
    def available_tools(self): return [t["name"] for t in self.task.get("available_tools", [])]

    def call_tool(self, name, args): raise NotImplementedError

    def reconcile_write(self, name, args, result):
        """ACTIVE READ-BACK (infrastructure): after a write, RE-READ the env to confirm the intended resource
        actually landed -- do not trust the write's own result envelope. Returns {confirmed: True|False|None,
        detail}. Default: this env cannot reconcile -> None (no-op; falls back to snapshot-based checks)."""
        return {"confirmed": None, "detail": "reconcile_unsupported"}

    def _healthy(self):
        """Best-effort runtime health of this env's backing service. Subclasses override with a real
        reachability check. Codex #10: a tool that is implemented+available but NOT healthy (service
        DOWN) means a failure is NOT the agent's fault and must not be scored as agent incompetence."""
        return True

    # Tools the runner has executable code for. Subclasses set their real surface; an empty
    # tuple means "trust available_tools" (back-compat: every offered tool counts as implemented).
    IMPLEMENTED_TOOLS = ()

    def capabilities(self):
        """Four-state capability manifest per tool (Codex #10): implemented (runner has code) /
        available (offered to this task) / authorized (allowed) / healthy (backing service up).

        PURPOSE: a tool that is implemented+available+authorized but NOT healthy means the backing
        SERVICE is down, so a resulting tool failure is NOT the agent's fault and must not be scored
        as agent incompetence. Read `healthy` before attributing an EnvironmentError to the agent."""
        deny = set(getattr(self, "_denied_tools", []) or [])
        healthy = self._healthy()
        impl = set(self.IMPLEMENTED_TOOLS)
        avail = set(self.available_tools())
        names = impl | avail if impl else avail  # empty IMPLEMENTED_TOOLS -> offered == implemented
        out = {}
        for n in sorted(names):
            is_impl = (n in impl) if impl else True
            out[n] = {
                "implemented": is_impl,
                "available": n in avail,
                "authorized": n not in deny,
                # only a tool actually backed by THIS env carries the live health signal
                "healthy": bool(healthy) if is_impl else False,
            }
        return out
    def teardown(self): ...

class FhirEnv(EnvironmentAdapter):
    type = "fhir"
    IMPLEMENTED_TOOLS = (
        "fhir_search", "fhir_read", "fhir_create", "get_lab_reference_range", "write_file",
        "fhir_patient_search_demographics", "fhir_condition_search_problems",
        "fhir_observation_search_labs", "fhir_observation_search_vitals",
        "fhir_observation_search_social_history", "fhir_medication_request_search_orders",
        "fhir_procedure_search_orders", "fhir_document_reference_search_clinical_notes",
        "fhir_service_request_search", "fhir_medication_request_create",
        "fhir_service_request_create", "fhir_communication_create_message", "fhir_appointment_create",
    )
    def __init__(self, task, fhir_base=None, workspace=None, aug_dir=None, **kw):
        super().__init__(task)
        self.base = fhir_base or os.environ.get("FHIR_BASE_URL") or self.cfg.get("default") or "http://localhost:38080/fhir"
        self.workspace = workspace or "/tmp/mh_workspace"
        self.aug_dir = aug_dir
        os.makedirs(self.workspace, exist_ok=True)
        self._lab = None
        self._health_cache = None
        # state digest (item #2): record the mutable resources THIS agent created in the sandbox,
        # so fhir_create/order/message changes state_summary() while read/search does not.
        self._created = []        # list of {resourceType,id,version,status} for created/updated resources
        self._written = []        # list of {path} for write_file deliverables (also mutable state)
    def _healthy(self):
        if self._health_cache is None:
            try:
                import urllib.request as _u
                _u.urlopen(_u.Request(self.base.rstrip("/") + "/metadata",
                                      headers={"Accept": "application/fhir+json"}), timeout=3)
                self._health_cache = True
            except Exception:
                self._health_cache = False
        return self._health_cache
    def _as_entries(self, res):
        """Flatten a FHIR search Bundle into {"entries":[resource,...], "total", "pages"} to match the
        upstream PhysicianBench tool output (eval_helpers/get_all_fhir_resources_from_trajectory expects
        an `entries` key; raw Bundle uses `entry` -> cp data-retrieval checks falsely fail)."""
        if isinstance(res, dict) and res.get("resourceType") == "Bundle":
            entries = [e["resource"] for e in res.get("entry", []) if isinstance(e, dict) and "resource" in e]
            return {"entries": entries, "total": res.get("total"), "pages": 1}
        return res
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
            res = self._get(f"/{rt}?" + urllib.parse.urlencode(params))
            return self._as_entries(res)  # match upstream PhysicianBench tool format {"entries":[...resources...]} (FHIR Bundle .entry -> flat list)
        if name == "fhir_read":
            return self._get(f"/{args['resourceType']}/{args['id']}")
        if name == "fhir_create":
            res = args.get("resource", args); rt = res.get("resourceType", "")
            data = json.dumps(res).encode()
            req = urllib.request.Request(f"{self.base}/{rt}", data=data, method="POST",
                                         headers={"Content-Type": "application/fhir+json"})
            try:
                out = json.load(urllib.request.urlopen(req, timeout=30))
                self._record_mutation(out, rt)  # mutable-state digest (item #2)
                return out
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
            content = args.get("content", "")
            open(fp, "w").write(content)
            # record deliverable as mutable state (path + content fingerprint) so re-writing the same
            # file with new content changes the digest, but a re-read of FHIR data does not.
            import hashlib as _hl
            self._written = [w for w in self._written if w.get("path") != rel]
            self._written.append({"path": rel, "sha": _hl.sha256(content.encode("utf-8", "replace")).hexdigest()[:12]})
            return {"written": fp}
        return {"error": f"unknown tool {name}"}

    def _record_mutation(self, out, rt):
        """Append/refresh the digest entry for a created/updated FHIR resource. Cheap, JSON-able,
        order-insensitive (state_summary sorts). Idempotent on (resourceType,id): a later version
        of the same resource replaces the earlier entry but bumps version -> hash changes."""
        try:
            if not isinstance(out, dict) or out.get("error"):
                return
            r_rt = out.get("resourceType", rt) or rt
            r_id = out.get("id")
            meta = out.get("meta") or {}
            ver = meta.get("versionId")
            # key status fields that callers can mutate (order/message lifecycle)
            status = out.get("status")
            entry = {"resourceType": r_rt, "id": r_id, "version": ver, "status": status}
            self._created = [c for c in self._created if not (c.get("resourceType") == r_rt and c.get("id") == r_id and r_id is not None)]
            self._created.append(entry)
        except Exception:
            pass

    def reconcile_write(self, name, args, result):
        """Active read-back: after fhir_create, GET the resource from the SERVER to confirm it persisted
        (not just that the POST returned 2xx); write_file -> confirm the file is on disk."""
        if name == "write_file":
            import os as _os
            wp = (result or {}).get("written")
            ok = bool(wp) and _os.path.exists(wp)
            return {"confirmed": bool(ok), "detail": ("file present: %s" % wp) if ok else "file not found on read-back"}
        if name != "fhir_create" or not isinstance(result, dict) or result.get("error"):
            return {"confirmed": None, "detail": "not_reconcilable"}
        _r = result.get("resource") if isinstance(result.get("resource"), dict) else result
        rid = _r.get("id"); rt = _r.get("resourceType") or (args.get("resource") or {}).get("resourceType")
        if not (rid and rt):
            return {"confirmed": None, "detail": "no_resource_id_in_result"}
        try:
            got = self._get("/%s/%s" % (rt, rid))
        except Exception as ex:
            return {"confirmed": False, "detail": "readback_error:%r" % (ex,)}
        if isinstance(got, dict) and not got.get("error") and str(got.get("id")) == str(rid):
            return {"confirmed": True, "detail": "%s/%s confirmed on server" % (rt, rid)}
        return {"confirmed": False, "detail": "%s/%s NOT found on server read-back" % (rt, rid)}

    def state_summary(self):
        """Deterministic, JSON-serializable digest of the MUTABLE sandbox state this agent can change
        (item #2). It reflects resources the agent created/ordered (fhir_*_create) and deliverables it
        wrote (write_file). A fhir_create / order / message / write_file MUTATES this; a fhir_search /
        fhir_read / get_lab_reference_range is read-only and leaves it unchanged. run.py._state_hash
        hashes this for state_record.state_before/after_hash. (Read-side FHIR server contents are NOT
        included: they are not agent-mutable within a task and reading them must not change the hash.)"""
        created = sorted(
            ({"resourceType": c.get("resourceType"), "id": c.get("id"),
              "version": c.get("version"), "status": c.get("status")} for c in self._created),
            key=lambda c: (str(c.get("resourceType")), str(c.get("id")), str(c.get("version"))))
        written = sorted(self._written, key=lambda w: str(w.get("path")))
        return {"created_resources": created, "deliverables": written}

class GuiEnvMock(EnvironmentAdapter):
    """HealthAdminBench v0: MOCK portal — browser actions mutate an in-memory full_state, which the
    JMESPath checkpoints score against. Proves the GUI substrate path without Playwright/npm/internet.
    v1 = drive the real NextJS portal via Playwright (harness/real_obs.py) and capture full_state.
    """
    type = "gui"
    IMPLEMENTED_TOOLS = ("navigate", "click", "type", "select", "upload", "submit", "snapshot")
    def __init__(self, task, **kw):
        super().__init__(task); self.full_state = None
    def reset(self):
        self.full_state = {"agentActions": {"viewedDenialDetails": False, "viewedRemittanceImage": False,
                                           "selectedDisposition": None, "documentedAppealInEpic": False},
                           "signals": {}, "triageNotes": "", "fields": {}, "_page": None}
    def call_tool(self, name, args):
        fs = self.full_state
        if name == "navigate":
            fs["_page"] = args.get("url") or args.get("target")
            import re as _re
            _m = _re.search(r"([A-Z]{2,5}-\d+)", str(fs["_page"] or ""))
            if _m: fs.setdefault("signals", {})["caseId"] = _m.group(1)   # displayed case
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

    def state_summary(self):
        """Deterministic, JSON-able digest of the GUI's mutable page/form state (item #2): current
        route, form field values, triage notes, and the set of completed actions (click/select/
        submit/upload). A navigate/type/select/submit MUTATES this; a `snapshot` is read-only and
        leaves it unchanged. None before reset(). run.py._state_hash hashes this for state_record."""
        fs = self.full_state
        if fs is None:
            return None
        return {
            "page": fs.get("_page"),
            "fields": {k: fs.get("fields", {})[k] for k in sorted(fs.get("fields", {}))},
            "triageNotes": fs.get("triageNotes", ""),
            "actions": {k: fs.get("agentActions", {})[k] for k in sorted(fs.get("agentActions", {}))},
            "signals": {k: fs.get("signals", {})[k] for k in sorted(fs.get("signals", {}))},
        }

class ToolSandboxEnv(EnvironmentAdapter):
    """MedCTA environment. mode=real (v1, DEFAULT): the 5 tools execute for real
    (ImageDescription / RegionAttributeDescription / OCR via local VLM backend; Calculator; GoogleSearch
    frozen corpus). mode=replay (v0): returns cached tool outputs from the reference trace so a
    ReplayAgent can validate ToolAcc/ArgAcc/Gacc WITHOUT a GPU. Select via MH_TOOL_MODE (real|replay).
    The image is resolved from task.context.images[0].path under MH_MEDCTA_IMG_ROOT.
    """
    type = "tool_sandbox"
    IMPLEMENTED_TOOLS = ("ImageDescription", "RegionAttributeDescription", "OCR", "Calculator", "GoogleSearch")
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
    def _healthy(self):
        return bool(self.image_path and os.path.exists(self.image_path))
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
    IMPLEMENTED_TOOLS = ("navigate", "click", "type", "select", "submit", "snapshot", "back", "scroll", "done", "download", "upload")

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
        self.workspace = os.environ.get("MH_GUI_WORKSPACE", "/tmp/hab_ws_%d" % os.getpid())
        self._last_download = None
        self._prev_hash = None  # NoProgress: detect API-success-but-no-state-change
    def _healthy(self):
        return self.page is not None

    def reset(self):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True, args=["--no-sandbox"])
        self.page = self._browser.new_page()
        # networkidle + settle: the portal's landing/list page is what calls initializeState() to seed
        # localStorage portals_state.emr. Detail pages (/emr/denied/[id]) only trackAction() and NO-OP if
        # state isn't seeded yet, so the start page MUST fully load+init before the agent drills in.
        self.page.goto(self.start_url, wait_until="domcontentloaded", timeout=45000)
        self._settle()
        self._read_state()
        self._init_obs = self._snap()

    def _settle(self):
        """Wait for the React render + the post-render trackAction()/initializeState() localStorage
        write to land. networkidle alone isn't enough (the state write happens in a useEffect AFTER the
        first paint), so add a fixed grace window."""
        try:
            self.page.wait_for_load_state("networkidle", timeout=8000)
        except Exception:
            pass
        self.page.wait_for_timeout(int(os.environ.get("MH_GUI_SETTLE_MS", "1200")))

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
        obs = self._observe()
        import hashlib
        h = hashlib.md5((self.page.url + obs).encode("utf-8", "ignore")).hexdigest()
        changed = self._prev_hash is not None and h != self._prev_hash  # API success != semantic progress
        self._prev_hash = h
        return {"ok": True, "url": self.page.url, "title": self.page.title(), "observation": obs,
                "state_changed": bool(changed), "surface_changed": bool(changed)}

    def reconcile_write(self, name, args, result):
        """Active read-back: after a `submit`, re-read the persisted EMR (localStorage portals_state.emr,
        the same object the checkpoints score) to confirm the submission actually registered -- catching a
        submit whose click timed out and left the case state unchanged (HAB-18). Only the commit (`submit`)
        is reconciled; reads / intermediate fills are not."""
        if name != "submit" or self.page is None:
            return {"confirmed": None, "detail": "not_reconcilable"}
        if isinstance(result, dict) and result.get("state_changed") is False:
            return {"confirmed": False, "detail": "submit left the persisted case state unchanged"}
        try:
            self._read_state()
            emr = self.full_state or {}
        except Exception as ex:
            return {"confirmed": False, "detail": "readback_error:%r" % (ex,)}
        if not emr:
            return {"confirmed": False, "detail": "EMR empty on read-back (nothing persisted)"}
        return {"confirmed": True, "detail": "case state persisted on read-back"}

    def state_summary(self):
        """Deterministic, JSON-able digest of the REAL portal's mutable state (item #2): the current
        route plus the persisted EMR (localStorage portals_state.emr, the same object the JMESPath
        checkpoints score). A submit/type/select that writes the EMR or changes the route MUTATES this;
        a `snapshot`/`scroll`/read-only navigate re-reads but does not change it. None before reset().
        Canonicalized via json.dumps(sort_keys) in run.py._state_hash, so key order is irrelevant."""
        if self.page is None and self.full_state is None:
            return None
        route = None
        try:
            import urllib.parse as _up
            route = _up.urlparse(self.page.url).path if self.page is not None else None
        except Exception:
            route = None
        return {"route": route, "emr": self.full_state or {}}

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
                self._settle()  # let the page's init/trackAction useEffect write localStorage
            elif name == "click":
                self._click(a)
            elif name == "type":
                self._fill(a, a.get("text", ""))
            elif name == "select":
                sel, _k = self._sel(a)
                val = a.get("value")
                try:
                    # native <select> element
                    self.page.select_option(sel, val, timeout=4000)
                except Exception:
                    # custom dropdown (e.g. the disposition picker is a <button> that opens a list of
                    # option buttons): click the control to open, then click the option whose text == value.
                    try:
                        self._click(a)
                        self.page.wait_for_timeout(300)
                    except Exception:
                        pass
                    self.page.get_by_text(str(val), exact=False).first.click(timeout=6000)
            elif name == "submit":
                if a.get("ref") or a.get("selector") or a.get("target"):
                    self._click(a)
                else:
                    self.page.click("button[type=submit]", timeout=8000)
            elif name == "snapshot":
                return self._snap()
            elif name == "back":                       # FAIRNESS: upstream HAB action parity
                self.page.go_back(wait_until="domcontentloaded", timeout=20000)
            elif name == "scroll":
                dy = -600 if str(a.get("direction", "down")).lower() == "up" else 600
                self.page.mouse.wheel(0, dy)
            elif name == "done":                       # explicit completion signal (maps to final)
                return {"done": True}
            elif name == "download":                   # FileAction: download -> workspace, record path+hash
                try:
                    with self.page.expect_download(timeout=20000) as _di:
                        self._click(a)
                    _dl = _di.value
                    os.makedirs(self.workspace, exist_ok=True)
                    _dest = os.path.join(self.workspace, _dl.suggested_filename or "download.bin")
                    _dl.save_as(_dest)
                    self._last_download = _dest
                    import hashlib
                    _h = hashlib.md5(open(_dest, "rb").read()).hexdigest()[:12]
                    _r = self._snap(); _r["downloaded"] = {"path": _dest, "hash": _h}; return _r
                except Exception as _e:
                    _r = self._snap(); _r["error"] = "download_failed: " + repr(_e)[:140]; return _r
            elif name == "upload":                     # FileAction: upload a (downloaded) file to an input
                try:
                    _fref = a.get("file_ref") or "last"
                    _fp = self._last_download if _fref == "last" else os.path.join(self.workspace, _fref)
                    _sel, _ = self._sel(a)
                    self.page.set_input_files(_sel, _fp, timeout=10000)
                    _r = self._snap(); _r["uploaded"] = {"path": _fp}; return _r
                except Exception as _e:
                    _r = self._snap(); _r["error"] = "upload_failed: " + repr(_e)[:140]; return _r
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
_FHIR_CANON_SEARCH = {"Patient":"fhir_patient_search_demographics","Condition":"fhir_condition_search_problems","MedicationRequest":"fhir_medication_request_search_orders","Procedure":"fhir_procedure_search_orders","DocumentReference":"fhir_document_reference_search_clinical_notes","ServiceRequest":"fhir_service_request_search"}
_FHIR_CANON_CREATE = {"MedicationRequest":"fhir_medication_request_create","ServiceRequest":"fhir_service_request_create","Communication":"fhir_communication_create_message","Appointment":"fhir_appointment_create"}
def _canon_fhir_tool(tool, args):
    """Map our generic fhir_search/read/create(resourceType=X) to the upstream PhysicianBench granular
    tool_name so native test_outputs.py checkpoints (which match metadata.tool_name) recognize the query.
    Conservative: a category-less Observation search -> labs only (never auto-credits vitals/social)."""
    a = args or {}
    if tool == "fhir_create":
        res = a.get("resource", a) or {}
        rt = res.get("resourceType", a.get("resourceType", ""))
        return _FHIR_CANON_CREATE.get(rt, tool)
    if tool in ("fhir_search", "fhir_read"):
        rt = a.get("resourceType", ""); cat = str(a.get("category", "")).lower()
        if rt == "Observation":
            if "vital" in cat: return "fhir_observation_search_vitals"
            if "social" in cat: return "fhir_observation_search_social_history"
            return "fhir_observation_search_labs"
        return _FHIR_CANON_SEARCH.get(rt, tool)
    return tool

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
        # Opt-in real Playwright portal. Two equivalent switches: MH_GUI_MODE=real (general) and
        # MH_HAB_REAL=1 (HealthAdminBench-specific alias used by the run command); either selects real.
        _real = os.environ.get("MH_GUI_MODE") == "real" or os.environ.get("MH_HAB_REAL") in ("1", "true", "True")
        cls = GuiEnvReal if _real else GuiEnvMock
        return cls(task, **kw)
    return ENV_REGISTRY[et](task, **kw)
