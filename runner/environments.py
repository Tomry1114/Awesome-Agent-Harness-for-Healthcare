"""Environment adapters — dispatch by unified task `environment.type`.

base: reset() / available_tools() / call_tool(name,args) / teardown().
PbNativeEnv (fhir) wraps the upstream PhysicianBench tools against the HAPI FHIR server; GuiEnvReal (gui) drives
the HealthAdminBench portal through Playwright; ToolSandboxEnv (tool_sandbox) executes the MedCTA tools.
ENV_REGISTRY maps environment.type -> adapter.
"""
import os, re, json, urllib.request, urllib.parse, urllib.error


def _safe_output_name(raw):
    """PB deliverable path -> a FLAT basename inside workspace/output.

    Every PhysicianBench checkpoint reads read_output_file(os.path.join(OUTPUT_DIR, "<name>")) with a FLAT
    name -- no upstream test reads a sub-path. Agents sometimes echo an absolute or sandbox path
    (/mnt/data/x.md, /tmp/x.md, /Users/Shared/x.md) or the conventional "workspace/output/x" prefix; keeping
    those sub-dirs would nest the file under workspace/output/<subdir>/ and the checkpoint's flat read would
    miss it (grades "Output file not found"). Collapsing to the basename lands the file exactly
    where the checkpoint looks and inherently blocks path traversal (basename can never escape).
    Returns None for a degenerate name (empty / '.' / '..').
    """
    rel = os.path.basename(os.path.normpath(raw or ""))
    if not rel or rel in (".", ".."):
        return None
    return rel

class EnvironmentAdapter:
    """BENCHMARK ADAPTER CONTRACT. Every benchmark registers ONE env class in ENV_REGISTRY
    and the runner drives it through this fixed surface — adding a benchmark = register a class, not
    edit the loop's execution path:
        reset()                      -> initialize episode (load_task handled by load_task() upstream)
        available_tools()            -> list[str] tools offered to the agent this task
        call_tool(name, args)        -> CanonicalResult-shaped dict (execute one action)
        capabilities()               -> four-state manifest {implemented/available/authorized/healthy}
        teardown()                   -> release resources
        initial_observation()        -> optional first CanonicalObservation (envs that have one)
    Source-metric evaluation (evaluate_source_metric) and artifact collection live in the
    scoring layer (scoring.run_checkpoint dispatch), so the contract spans env + scorer. Conformance is
    enforced by tests/test_conformance.test_benchmark_adapter_contract over every ENV_REGISTRY class."""
    type = "base"
    # {tool: {"path": arg, "content": arg}} -- the tools whose EXECUTION writes the task's deliverable, and
    # which argument carries each part. Declared here because the adapter is what performs the write. None =
    # this substrate declares no written deliverable, so "is the deliverable absent" is UNKNOWN on it, never
    # guessed. Read by the runtime (kernel) through harness.substrate_catalog.deliverable_writes.
    DELIVERABLE_WRITES = None
    # The tools with which the agent ENDS the episode on this substrate, as the upstream benchmark defines them
    # (HealthAdminBench done(): harness/environment.py step sets done and the episode ends). run.py applies such a call
    # as the agent's final (the shell's one episode-end path) instead of executing it. () = the agent ends only by
    # answering.
    EPISODE_END_TOOLS = ()
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
        reachability check. A tool that is implemented+available but NOT healthy (service
        DOWN) means a failure is NOT the agent's fault and must not be scored as agent incompetence."""
        return True

    # Tools the runner has executable code for. Subclasses set their real surface; an empty
    # tuple means "trust available_tools" (every offered tool counts as implemented).
    IMPLEMENTED_TOOLS = ()

    def capabilities(self):
        """Four-state capability manifest per tool: implemented (runner has code) /
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

_PB_UPSTREAM = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "benchmark", "PhysicianBench", "PhysicianBench")


class PbNativeEnv(EnvironmentAdapter):
    """PB-native parity env -- the ONLY FHIR env. Wraps the UPSTREAM PhysicianBench ToolRegistry (full JSON
    schemas, real Bundle.link[next] pagination, code/date/count/page_limit filters, structured create
    helpers). Also carries the contract machinery (capabilities via IMPLEMENTED_TOOLS, reconcile_write,
    state digest, workspace write_file, fhir_read/lab-ref)."""
    type = "fhir"
    # call_tool("write_file") below is the one place a PB deliverable is written (self._written), and it
    # reads exactly these two arguments. A fhir_*_create is a created RESOURCE, not the deliverable, however
    # long its note_text (e.g. a long ServiceRequest note is not a deliverable; treating it as one would keep
    # committed_without_artifact from firing for the rest of the episode).
    # mode/overwrite: upstream file_tools.write_file(file_path, content, mode="w") overwrites when mode == "w" and appends
    # for any other mode; the kernel's record of the deliverable follows the same rule.
    # path: the parameter the tool declares (tool_descriptions/write_file.yaml, upstream file_path), not `path`, the
    # workspace handler's internal name below; naming `path` here would leave the kernel's last_write.path None for every
    # native write.
    DELIVERABLE_WRITES = {"write_file": {"path": "file_path", "content": "content", "mode": "mode", "overwrite": "w"}}
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
        # state digest: record the mutable resources THIS agent created in the sandbox,
        # so fhir_create/order/message changes state_summary() while read/search does not.
        self._created = []        # list of {resourceType,id,version,status} for created/updated resources
        self._written = []        # list of {path} for write_file deliverables (also mutable state)
        self._setup_native_registry()

    def _setup_native_registry(self):
        import sys, functools, inspect
        if _PB_UPSTREAM not in sys.path:
            sys.path.insert(0, _PB_UPSTREAM)
        from agent.tool_registry import (ToolRegistry, register_all_tools, FHIR_TOOL_SCHEMAS, FILE_TOOL_SCHEMAS)
        reg = ToolRegistry()
        register_all_tools(reg)
        for nm in list(reg._tools.keys()):
            fn, sch = reg._tools[nm]
            if "base_url" in inspect.signature(fn).parameters:
                reg._tools[nm] = (functools.partial(fn, base_url=self.base), sch)
        self._reg = reg
        self._native_specs = [{"name": sch["name"], "signature": (sch.get("description") or "")[:120],
                               "description": sch.get("description", ""), "parameters": sch.get("parameters", {})}
                              for sch in (FHIR_TOOL_SCHEMAS + FILE_TOOL_SCHEMAS)]
        self.task["available_tools"] = self._native_specs

    def available_tools(self):
        return [t["name"] for t in self._native_specs]
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
        # write_file: map native file_path -> path for the workspace-aware handler below.
        if name == "write_file" and isinstance(args, dict) and "file_path" in args and "path" not in args:
            args = dict(args); args["path"] = args.pop("file_path")
        # Faithful UPSTREAM tools (paginated/filtered searches + structured creates) -> registry dispatch.
        if name != "write_file" and name in self._reg._tools:
            try:
                out = self._reg.dispatch(name, dict(args or {}))
            except Exception as ex:
                return {"error": "native_dispatch_error:%r" % ex}
            if name.endswith("_create") and isinstance(out, dict) and out.get("id") and out.get("resourceType") != "OperationOutcome":
                self._record_mutation(out, out.get("resourceType") or "")
            return out
        if name == "fhir_search":   # generic fallback (internal callers only); agents use the upstream granular tools
            rt = args.get("resourceType", ""); params = {k: v for k, v in args.items() if k != "resourceType"}
            params = self._normalize_search(rt, params)
            return self._as_entries(self._get(f"/{rt}?" + urllib.parse.urlencode(params)))
        if name == "fhir_read":
            return self._get(f"/{args['resourceType']}/{args['id']}")
        if name == "fhir_create":
            res = args.get("resource", args); rt = res.get("resourceType", "")
            data = json.dumps(res).encode()
            req = urllib.request.Request(f"{self.base}/{rt}", data=data, method="POST",
                                         headers={"Content-Type": "application/fhir+json"})
            try:
                out = json.load(urllib.request.urlopen(req, timeout=30))
                self._record_mutation(out, rt)  # mutable-state digest
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
            # Collapse the deliverable to a FLAT basename inside workspace/output. PB checkpoints read a flat
            # OUTPUT_DIR/<name> (no test reads a sub-path); an absolute/sandbox path or a workspace/output/
            # prefix would otherwise nest the file and the flat read would miss it. Also blocks traversal.
            raw = args.get("path", "out.txt")
            rel = _safe_output_name(raw)
            if rel is None:
                return {"error": "path traversal rejected"}
            fp = os.path.join(self.workspace, rel)
            os.makedirs(os.path.dirname(fp) or self.workspace, exist_ok=True)
            content = args.get("content", "")
            # upstream PhysicianBench tools/file_tools.py write_file: mode "w" (the default) overwrites, any other
            # mode appends (so an append keeps the earlier part of the file)
            with open(fp, "w" if args.get("mode", "w") == "w" else "a") as _fh:
                _fh.write(content)
            with open(fp, encoding="utf-8", errors="replace") as _fh:
                _file_text = _fh.read()
            # record deliverable as mutable state (path + fingerprint of the file's content) so re-writing the same
            # file with new content changes the digest, but a re-read of FHIR data does not.
            import hashlib as _hl
            self._written = [w for w in self._written if w.get("path") != rel]
            self._written.append({"path": rel, "sha": _hl.sha256(_file_text.encode("utf-8", "replace")).hexdigest()[:12]})
            out = {"written": fp, "path": rel}
            if os.path.basename(os.path.normpath(raw)) != raw.strip():
                out["normalized_from"] = raw  # agent gave a sub-path/absolute path; flattened to basename
            return out
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
        # reconcile BOTH the generic fhir_create AND the native fhir_*_create tools (the native
        # ones return the raw resource with an id) -- else a successful native create is recorded UNKNOWN.
        _NATIVE_CREATE = {"fhir_service_request_create": "ServiceRequest", "fhir_medication_request_create": "MedicationRequest",
                          "fhir_communication_create_message": "Communication", "fhir_appointment_create": "Appointment"}
        if (name != "fhir_create" and name not in _NATIVE_CREATE) or not isinstance(result, dict) or result.get("error") or result.get("resourceType") == "OperationOutcome":
            return {"confirmed": None, "detail": "not_reconcilable"}
        _r = result.get("resource") if isinstance(result.get("resource"), dict) else result
        rid = _r.get("id"); rt = _r.get("resourceType") or (args.get("resource") or {}).get("resourceType") or _NATIVE_CREATE.get(name)
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
        """Deterministic, JSON-serializable digest of the MUTABLE sandbox state this agent can change.
        It reflects resources the agent created/ordered (fhir_*_create) and deliverables it
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


MEDCTA_IMG_ROOT = os.path.join(os.path.dirname(__file__), "..", "benchmark", "MedCTA", "opencompass", "data",
                               "medcta_dataset")


class ToolSandboxEnv(EnvironmentAdapter):
    """MedCTA environment: the 5 tools execute for real
    (ImageDescription / RegionAttributeDescription / OCR through the VLM backend; Calculator; GoogleSearch over
    the Serper API, filtered by runner/search_guard.py). The image is resolved from task.context.images[0].path under MEDCTA_IMG_ROOT.
    """
    type = "tool_sandbox"
    IMPLEMENTED_TOOLS = ("ImageDescription", "RegionAttributeDescription", "OCR", "Calculator", "GoogleSearch")
    GUARDED_SEARCH_TOOLS = ("GoogleSearch",)   # web search behind runner/search_guard.py; run.py counts its blocks
    def __init__(self, task, **kw):
        super().__init__(task)
        self.image_path = self._resolve_image(task)
    def _resolve_image(self, task):
        imgs = (task.get("context") or {}).get("images") or []
        if not imgs: return None
        rel = imgs[0].get("path") or ""
        fp = os.path.join(MEDCTA_IMG_ROOT, rel)
        return fp if os.path.exists(fp) else rel
    def _healthy(self):
        return bool(self.image_path and os.path.exists(self.image_path))
    def call_tool(self, name, args):
        import adapters.medcta.tools_medcta as tools_medcta
        # the task id selects this item's provenance rules in the search guard (runner/search_guard.py)
        out = tools_medcta.execute(name, args or {}, self.image_path, task_id=self.task.get("task_id"))
        res = {"tool": name, "args": args, "output": out["output"], "mode": "real"}
        if out.get("blocked"):
            res["blocked"] = out["blocked"]   # the search guard refused the query or withheld results
        return res


_HAB_AXTREE_FN = None
GUI_WORKSPACE = "/tmp/hab_ws_%d"   # per-process download workspace, % pid
GUI_SETTLE_MS = 1200               # fixed grace window after networkidle (GuiEnvReal._settle)


def _gui_axtree(page):
    """Upstream-faithful GUI observation: the merged accessibility tree from
    benchmark/HealthAdminBench/harness/real_obs.build_axtree_text -- CDP Accessibility.getFullAXTree per
    frame, merged across ALL iframes, flattened to `[testid] role "name"` lines. This is the EXACT
    extractor upstream HAB feeds its agent (environment.py:412). Loaded via importlib-by-path because the
    upstream package is also named `harness` and would collide with runner/harness. real_obs.py is
    self-contained (re/time/typing/loguru/playwright only), so a standalone spec load is safe."""
    global _HAB_AXTREE_FN
    if _HAB_AXTREE_FN is None:
        import importlib.util
        _rp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "benchmark", "HealthAdminBench", "harness", "real_obs.py")
        _spec = importlib.util.spec_from_file_location("hab_real_obs", _rp)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _HAB_AXTREE_FN = _mod.build_axtree_text
    return _HAB_AXTREE_FN(page)


class BrowserPageCrashed(BaseException):
    """The portal page's renderer died mid-episode, so the episode is not a measurement.

    A Playwright call in flight when the renderer dies never returns (a pending page.evaluate keeps waiting
    after the "crash" event has fired), which would stall the batch for tens of minutes. GuiEnvReal closes the
    page on "crash", which fails the waiting call, and raises this on its next use of the page. A BaseException
    so the step loop's `except Exception` handlers, which turn a tool's failure into feedback for the agent,
    cannot hand a dead browser to the agent as if it were the task; run.run_task turns it into a crash result
    (void, re-run)."""


# REGISTERED DEVIATION: browser action timeouts are 8s here, where the upstream harness uses
# settings.browser.timeout_seconds=15 (and file_timeout_seconds=10). The value is held fixed so that all
# agent models are measured under one setting. Known consequence: a click on a control that is briefly
# non-actionable (a modal still animating open) can time out here where upstream would have waited it out.
class GuiEnvReal(EnvironmentAdapter):
    """HealthAdminBench v1: drives the REAL Next.js portal via Playwright. Gives the agent a readable
    OBSERVATION each step: the UPSTREAM merged accessibility tree (harness/real_obs.build_axtree_text --
    CDP Accessibility.getFullAXTree per frame, merged across iframes, flattened to `[testid] role "name"`
    lines; the exact extractor upstream HAB feeds its agent). The agent addresses elements by `ref` = the
    bracket id (= element data-testid); click/type/select accept {"ref": "<testid>"}; a raw CSS selector
    or visible text still works as fallback.
    full_state is read from the portal's localStorage key portals_state -> .emr (what the JMESPath
    checkpoints score). Requires the portal at MH_PORTAL_BASE (default http://localhost:3002) and
    chromium."""
    type = "gui"
    # upstream harness/environment.py step: done() sets done and ends the episode; run.py applies a `done` call as
    # the agent's final answer instead of executing it.
    EPISODE_END_TOOLS = ("done",)
    # Agent-facing surface is the paper's Table 5 set (see convert.py GUI_TOOLS). navigate/submit/snapshot
    # stay IMPLEMENTED because the harness recovery substrate calls them directly, but they are not
    # advertised to the agent.
    IMPLEMENTED_TOOLS = ("navigate", "click", "type", "fill", "select", "press", "submit", "snapshot",
                         "back", "scroll", "done", "download", "upload")

    def __init__(self, task, **kw):
        super().__init__(task)
        import urllib.parse as up
        self.base = os.environ.get("MH_PORTAL_BASE", "http://localhost:3002")
        self.start_url = self._resolve_start_url()
        self._pw = self._browser = self.page = None
        self.full_state = None
        self._portals = {}
        self._init_obs = None
        self.workspace = GUI_WORKSPACE % os.getpid()
        self._last_download = None
        self._prev_hash = None  # NoProgress: detect API-success-but-no-state-change
    # env_paths mirrors native harness/config/settings.py env_paths (portal id -> app root path).
    _ENV_PATHS = {"emr": "/emr", "epic": "/emr", "payer_a": "/payer-a", "payer-a": "/payer-a",
                  "payer_b": "/payer-b", "payer-b": "/payer-b", "fax_portal": "/fax-portal", "fax-portal": "/fax-portal"}

    @classmethod
    def _portal_root(cls, website_id):
        return cls._ENV_PATHS.get(str(website_id or "").strip().lower(), "/emr")

    def _resolve_start_url(self):
        """Native HAB _build_start_url() resolution, rebased onto the local portal host (MH_PORTAL_BASE).
        Native website.url is only the PORTAL ROOT (e.g. https://.../ for prior_auth/dme); the real deep
        link lives in config.start_url (/worklist, /emr/dme). Rule: absolute -> rebase host; path already
        under an explicit portal root (/emr,/payer-a,...) -> base+path; else prepend the website.id portal
        root (epic/emr -> /emr). Falls back to website.url path when no native_config (denial carries the
        deep path there). This lands prior_auth/dme on /emr/worklist // /emr/dme rather than on '/'."""
        import urllib.parse as _up
        base = self.base.rstrip("/")
        site = self.cfg.get("website") or {}
        nc = self.cfg.get("native_config") or {}
        explicit = ("/emr", "/payer-a", "/payer-b", "/fax-portal")
        su = nc.get("start_url")
        if su:
            su = su.replace("{{TASK_ID}}", str(nc.get("task_id") or "")).replace("{{RUN_ID}}", "")
            pr = _up.urlparse(su)
            if pr.scheme in ("http", "https"):
                return base + (pr.path or "/")
            su = pr.path or su  # drop any tracking query
            if su.startswith(explicit):
                return base + su
            return base + self._portal_root(site.get("id")) + su
        # fallback: derive from website.url path (denial: /denied)
        path = _up.urlparse(site.get("url") or "").path or "/"
        if path.startswith(explicit):
            return base + path
        root = self._portal_root(site.get("id"))
        return base + root + ("" if path == "/" else path)

    def _healthy(self):
        return self.page is not None

    def reset(self):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True, args=["--no-sandbox"])
        self.page = self._browser.new_page()
        self.page_crashed = False
        self.page.on("crash", self._on_page_crash)
        # networkidle + settle: the portal's landing/list page is what calls initializeState() to seed
        # localStorage portals_state.emr. Detail pages (/emr/denied/[id]) only trackAction() and NO-OP if
        # state isn't seeded yet, so the start page MUST fully load+init before the agent drills in.
        self.page.goto(self.start_url, wait_until="domcontentloaded", timeout=45000)
        self._settle()
        self._read_state()
        self._init_obs = self._snap()

    def _on_page_crash(self, page):
        """The renderer died: a call waiting on it never returns, and closing the page fails it (BrowserPageCrashed)."""
        self.page_crashed = True
        try:
            page.close()
        except Exception:
            pass

    def _alive(self):
        if getattr(self, "page_crashed", False):
            raise BrowserPageCrashed("the portal page's renderer crashed during the episode")

    def _settle(self):
        """Wait for the React render + the post-render trackAction()/initializeState() localStorage
        write to land. networkidle alone isn't enough (the state write happens in a useEffect AFTER the
        first paint), so add a fixed grace window."""
        try:
            self.page.wait_for_load_state("networkidle", timeout=8000)
        except Exception:
            pass
        self.page.wait_for_timeout(GUI_SETTLE_MS)

    def initial_observation(self):
        return self._init_obs

    def _read_state(self):
        self._alive()
        try:
            raw = self.page.evaluate("() => { try { return localStorage.getItem('portals_state'); } catch(e){ return null; } }")
            st = json.loads(raw) if raw else None
            st = st if isinstance(st, dict) else {}
            self._portals = {k: (st.get(k) if isinstance(st.get(k), dict) else {}) for k in ("emr", "payerA", "payerB", "fax")}
            self.full_state = self._portals.get("emr") or {}
        except Exception:
            pass
        return self.full_state

    def restore_persisted_content(self, lost):
        """Content-preservation guard restore (harness recovery): re-persist ROOT-agent-authored text that a
        recovery navigation cleared as a portal side effect (e.g. tab-switch wipes triageNotes). Writes the
        agent's OWN prior value back into localStorage portals_state ONLY where the leaf is now empty/absent
        -- additive, idempotent, oracle-blind, NEVER new content. Returns #fields restored."""
        if not lost or self.page is None:
            return 0
        TOP = {"full_state": "emr", "payer_a_state": "payerA", "payer_b_state": "payerB", "fax": "fax",
               "aetna_state": "payerA", "anthem_state": "payerB"}
        payload = {}
        for dotted, val in lost.items():
            segs = str(dotted).split(".")
            top = TOP.get(segs[0])
            if not top or len(segs) < 2:
                continue
            payload.setdefault(top, []).append({"path": segs[1:], "value": val})
        if not payload:
            return 0
        js = """(payload) => {
          let st; try { st = JSON.parse(localStorage.getItem('portals_state')||'{}'); } catch(e){ st={}; }
          if (typeof st !== 'object' || st === null) st = {};
          let n = 0;
          for (const top in payload) {
            st[top] = st[top] || {};
            for (const item of payload[top]) {
              let node = st[top]; const p = item.path;
              for (let i=0; i<p.length-1; i++){ node[p[i]] = node[p[i]] || {}; node = node[p[i]]; }
              const leaf = p[p.length-1]; const cur = node[leaf];
              if (cur === undefined || cur === null || cur === '') { node[leaf] = item.value; n++; }
            }
          }
          localStorage.setItem('portals_state', JSON.stringify(st));
          return n;
        }"""
        try:
            n = self.page.evaluate(js, payload)
            if n:
                self._read_state()
            return n or 0
        except Exception:
            return 0

    def eval_state(self):
        """Native HAB get_final_state() eval root assembled from ALL portals in localStorage
        (portals_state: emr/payerA/payerB/fax) -- NOT just emr. Makes payer_a_state / payer_b_state /
        aetna_state / anthem_state / full_state.faxPortal checkpoints reachable. Generic hook the
        scorer roots JMESPath queries at; envs without it fall back to bare full_state. The view is the shell's one
        obligation-evaluation view (harness.obligation_gate.obligation_view), the one recovery evaluates too."""
        self._alive()
        from harness.obligation_gate import obligation_view
        self._read_state()
        portals = getattr(self, "_portals", None) or {}
        return obligation_view({"emr": portals.get("emr") or (self.full_state or {}), "payerA": portals.get("payerA"),
                                "payerB": portals.get("payerB"), "fax": portals.get("fax")})

    def _observe(self):
        self._alive()
        try:
            ax = _gui_axtree(self.page)
        except Exception as e:
            return "(observation failed: %r)" % e
        # Upstream-faithful agent-facing format: URL + the merged-axtree PAGE ELEMENTS block
        # (upstream prompts.py:587). Each line's bracket id == element data-testid -> pass it as the
        # `ref` arg to click/type/select. Full a11y tree including iframes.
        return "URL: %s\nPAGE ELEMENTS (use the identifier shown in [brackets] as ref):\n%s" % (self.page.url, ax)

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
        """Active read-back after a `submit`: re-read the persisted 4-portal state and confirm an EXACT
        postcondition the caller/adapter supplies instead of the coarse "the EMR merely changed"
        heuristic (which FALSE-POSITIVES on any EMR write and FALSE-NEGATIVES on payer/fax submits that never
        touch the EMR). The adapter says WHERE the spec lives: args["expected_postcondition"] /
        args["_postcondition"], or the env attribute self._expected_postcondition (so BOTH the executor path
        -- env.reconcile_write(tool,args,res) -- and the substrate path -- driver.reconcile_write("submit",{},r)
        -- pick it up). Spec = a dotted PORTAL path (emr/payerA/payerB/fax.<...>; a path not starting with a
        known portal is rooted at emr) + ONE predicate: equals / is_true / non_empty / min_length / min_value.
        Examples: {"path":"payerA.submissions","non_empty":true},
        {"path":"payerB.appealActions.submittedAppeal","is_true":true},
        {"path":"fax.faxesSent","min_value":N}, {"path":"agentActions.selectedDisposition","non_empty":true}.
        With NO spec (or an unevaluable one) it FALLS BACK to the coarse overall-changed check, FLAGGED
        fallback=True. Only the commit (`submit`) is reconciled; reads / intermediate fills are not."""
        self._alive()
        if name != "submit" or self.page is None:
            return {"confirmed": None, "detail": "not_reconcilable"}
        spec = None
        if isinstance(args, dict):
            spec = args.get("expected_postcondition") or args.get("_postcondition")
        if spec is None:
            spec = getattr(self, "_expected_postcondition", None)
        try:
            self._read_state()
        except Exception as ex:
            return {"confirmed": False, "detail": "readback_error:%r" % (ex,)}
        # (a) EXACT postcondition supplied -> verify precisely against the freshly-read portal state.
        if isinstance(spec, dict) and spec.get("path"):
            ok = self._eval_postcondition(spec)
            if ok is True:
                return {"confirmed": True, "postcondition": dict(spec),
                        "detail": "postcondition satisfied on read-back: %s" % spec.get("path")}
            if ok is False:
                return {"confirmed": False, "postcondition": dict(spec),
                        "detail": "postcondition NOT satisfied on read-back: %s" % spec.get("path")}
            # ok is None -> spec present but not evaluable here; fall through to the FLAGGED heuristic.
        # (b) FALLBACK: no exact spec (or unevaluable) -> coarse overall-changed heuristic, FLAGGED.
        if isinstance(result, dict) and result.get("state_changed") is False:
            return {"confirmed": False, "fallback": True,
                    "detail": "submit left the persisted case state unchanged"}
        emr = self.full_state or {}
        if not emr:
            return {"confirmed": False, "fallback": True,
                    "detail": "EMR empty on read-back (nothing persisted)"}
        return {"confirmed": True, "fallback": True,
                "detail": "case state persisted on read-back (no exact postcondition)"}

    def _resolve_portal_path(self, path):
        """Resolve a dotted path against the freshly-read 4-portal state (self._portals, populated by
        _read_state). The first segment MAY name a portal (emr/payerA/payerB/fax); a path that does NOT
        start with a known portal is rooted at emr (so emr-relative markers like
        'agentActions.selectedDisposition' still resolve). Numeric segments index lists. Leaf or None."""
        portals = getattr(self, "_portals", None) or {}
        segs = [s for s in str(path).split(".") if s != ""]
        if not segs:
            return None
        if segs[0] in ("emr", "payerA", "payerB", "fax"):
            node = portals.get(segs[0]); segs = segs[1:]
        else:
            node = portals.get("emr") or (self.full_state or {})
        for s in segs:
            if isinstance(node, dict):
                node = node.get(s)
            elif isinstance(node, list):
                try:
                    node = node[int(s)]
                except Exception:
                    return None
            else:
                return None
        return node

    def _eval_postcondition(self, spec):
        """Evaluate ONE exact postcondition predicate against self._portals. Returns True (met) / False
        (refuted) / None (unevaluable -> caller falls back). Predicate key present in the spec dict:
        equals (==), is_true (bool/'true'), non_empty (list/dict/str/truthy), min_length (len>=N),
        min_value (numeric>=N). Substrate-generic: the HAB adapter builds the spec, this only reads state."""
        if not isinstance(spec, dict) or not spec.get("path"):
            return None
        node = self._resolve_portal_path(str(spec.get("path")))
        if "equals" in spec:
            return node == spec.get("equals")
        if spec.get("is_true"):
            return node is True or (isinstance(node, str) and node.strip().lower() == "true")
        if spec.get("non_empty"):
            if isinstance(node, str):
                return bool(node.strip())
            if isinstance(node, (list, dict)):
                return len(node) > 0
            return node is not None and node is not False and node != 0
        if "min_length" in spec:
            try:
                return isinstance(node, (list, str, dict)) and len(node) >= int(spec.get("min_length"))
            except Exception:
                return None
        if "min_value" in spec:
            try:
                return (isinstance(node, (int, float)) and not isinstance(node, bool)
                        and float(node) >= float(spec.get("min_value")))
            except Exception:
                return None
        return None

    def list_controls(self):
        """Live control model (ref/role/label/name/required/value) for the PreCommitAudit -- same shape the
        recovery driver uses. required = HTML required/aria-required (the asterisk convention is handled by
        the caller via the label). Benchmark-neutral; empty on any error."""
        if self.page is None:
            return []
        from adapters.hab.gui_control_model import CONTROL_JS
        try:
            r = self.page.evaluate(CONTROL_JS)
            return r if isinstance(r, list) else []
        except Exception:
            return []

    def state_summary(self):
        """Deterministic, JSON-able digest of the REAL portal's mutable state: the current
        route plus ALL FOUR persisted portals (localStorage portals_state.emr/payerA/payerB/fax -- the
        same objects the JMESPath checkpoints score across EMR + payer + fax portals). A submit/type/select that writes the EMR or changes the route MUTATES this;
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
        _portals = getattr(self, "_portals", None) or {}
        return {"route": route, "emr": self.full_state or (_portals.get("emr") or {}),
                "payerA": _portals.get("payerA") or {}, "payerB": _portals.get("payerB") or {},
                "fax": _portals.get("fax") or {}}

    def _sel(self, a):
        if a.get("ref") not in (None, ""):
            # Upstream resolution: bracket id == element data-testid (real_obs flatten emits testid as bid).
            return ('[data-testid="%s"]' % a.get("ref")), "css"
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

    def _write_receipt(self, a, attempted):
        """Read the TARGET control back after a type/select so the caller can tell a landed write from a
        no-op / mis-route. Oracle-blind; never gold."""
        ref = a.get("ref")
        if not ref or self.page is None:
            return None
        try:
            from adapters.hab.gui_control_model import READBACK_JS, write_landed
            rb = self.page.evaluate(READBACK_JS, str(ref))
        except Exception:
            return None
        if not rb or not rb.get("found"):
            return {"ref": ref, "field": a.get("field_intent") or a.get("field"), "attempted": str(attempted)[:80],
                    "observed": None, "landed": False, "reason": "control_not_found"}
        observed = rb.get("selected_text") or rb.get("value") or ("checked" if rb.get("checked") else "")
        return {"ref": ref, "field": a.get("field_intent") or a.get("field"), "control_label": rb.get("label"),
                "control_section": rb.get("section"),
                "attempted": str(attempted)[:80], "observed": str(observed)[:80],
                "landed": bool(write_landed(attempted, rb))}

    def _select_custom(self, trigger_sel, val):
        """Custom (non-native) dropdown: open the trigger, then pick the option WITHIN the trigger's OWN popup
        by EXACT name -- never a whole-page fuzzy first-match (which can land on body text / another dropdown /
        a hidden option). Popup = aria-controls/aria-owns target, else the last visible listbox/menu. Falls
        back to a PAGE-scoped EXACT match only (never contains-.first). Returns True if a click
        was issued (the write_receipt readback is the authoritative landed check)."""
        try:
            trigger = self.page.locator(trigger_sel).first
            try:
                trigger.scroll_into_view_if_needed(timeout=1500)
            except Exception:
                pass
            trigger.click(timeout=4000)
            self.page.wait_for_timeout(250)
            pid = None
            try:
                pid = trigger.get_attribute("aria-controls") or trigger.get_attribute("aria-owns")
            except Exception:
                pid = None
            scope = None
            if pid:
                cand = self.page.locator("#%s" % pid)
                try:
                    if cand.count() > 0:
                        scope = cand
                except Exception:
                    scope = None
            if scope is None:
                try:
                    vis = self.page.locator("[role=listbox]:visible, [role=menu]:visible")
                    if vis.count() > 0:
                        scope = vis.last
                except Exception:
                    scope = None
            root = scope if scope is not None else self.page
            v = str(val)
            for getter in (
                lambda: root.get_by_role("option", name=v, exact=True),
                lambda: root.get_by_role("menuitem", name=v, exact=True),
                lambda: root.get_by_role("button", name=v, exact=True),
                lambda: root.get_by_text(v, exact=True),
            ):
                try:
                    opt = getter().first
                    if opt.count() > 0:
                        opt.click(timeout=3000)
                        return True
                except Exception:
                    continue
            return False
        except Exception:
            return False

    def call_tool(self, name, args):
        self._alive()
        a = dict(args or {})
        # The element id may arrive under ref/target/selector/field -- models do not always honor the
        # documented `ref` (some put it in `target`). A BARE identifier token IS the axtree bracket id
        # (== element data-testid); normalize it into a["ref"] so _sel / write_receipt / download all
        # resolve it uniformly as [data-testid="..."]. CSS selectors (#/./[) and multi-word text are left
        # untouched (not bare tokens) so their existing selector/text paths still apply. navigate ignores
        # ref, so a coerced ref there is harmless.
        if not a.get("ref"):
            for _k in ("id", "target", "selector", "field"):
                _v = a.get(_k)
                if isinstance(_v, str) and re.match(r"^[A-Za-z0-9_\-]+$", _v.strip()):
                    a["ref"] = _v.strip(); break
        try:
            if name == "navigate":
                tgt = a.get("url") or a.get("target") or "/"
                import urllib.parse as _up
                pth = _up.urlparse(tgt).path if "://" in tgt else (tgt if tgt.startswith("/") else "/" + tgt)
                self.page.goto(self.base + (pth or "/"), wait_until="domcontentloaded", timeout=45000)
                self._settle()  # let the page's init/trackAction useEffect write localStorage
            elif name == "click":
                self._click(a)
            elif name in ("type", "fill"):          # paper Table 5 calls this fill(id, text)
                self._fill(a, a.get("text", ""))
                try: self.page.wait_for_timeout(400)   # readback of the target control
                except Exception: pass
                _r = self._snap(); _wr = self._write_receipt(a, a.get("text", ""))
                if _wr is not None: _r["write_receipt"] = _wr
                return _r
            elif name == "select":
                sel, _k = self._sel(a)
                val = a.get("option", a.get("value"))   # paper Table 5: select(id, option)
                try:
                    self.page.select_option(sel, val, timeout=3000)   # native <select>
                except Exception:
                    self._select_custom(sel, val)                     # scoped, exact, popup-only pick
                try: self.page.wait_for_timeout(400)   # readback of the target control
                except Exception: pass
                _r = self._snap(); _wr = self._write_receipt(a, val)
                if _wr is not None: _r["write_receipt"] = _wr
                return _r
            elif name == "submit":
                if a.get("ref") or a.get("selector") or a.get("target"):
                    self._click(a)
                else:
                    self.page.click("button[type=submit]", timeout=8000)
            elif name == "snapshot":
                return self._snap()
            elif name == "press":                      # paper Table 5: press(id, key)
                _sel_p, _ = self._sel(a)
                self.page.press(_sel_p, str(a.get("key") or "Enter"), timeout=8000)
            elif name == "back":                       # upstream HAB action parity
                self.page.go_back(wait_until="domcontentloaded", timeout=20000)
            elif name == "scroll":
                dy = -600 if str(a.get("direction", "down")).lower() == "up" else 600
                self.page.mouse.wheel(0, dy)
            elif name == "done":                       # episode end (EPISODE_END_TOOLS): run.py applies it as the
                return {"done": True}                  # final; reached only by a caller other than the loop
            elif name == "download":                   # FileAction: download -> workspace, record path+hash
                _dref = str(a.get("ref") or a.get("target") or a.get("selector") or "")
                _reg = self.__dict__.setdefault("_download_registry", {})
                if _dref and _dref in _reg:             # IDEMPOTENT: already downloaded -> no re-click, no new
                    _c = _reg[_dref]                     # file, no persisted-state change. Progress is by content,
                    self._last_download = _c["path"]     # not file count -> the StagnationGuard sees no progress.
                    _r = self._snap(); _r["state_changed"] = False; _r["surface_changed"] = False
                    _r["downloaded"] = {"path": _c["path"], "hash": _c["hash"]}
                    _r["already_downloaded"] = True; _r["cached"] = True
                    return _r
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
                    if _dref:
                        _reg[_dref] = {"path": _dest, "hash": _h}
                    _r = self._snap(); _r["downloaded"] = {"path": _dest, "hash": _h}; return _r
                except Exception as _e:
                    _r = self._snap(); _r["error"] = "download_failed: " + repr(_e)[:140]; return _r
            elif name == "upload":                     # FileAction: upload a (downloaded) file to an input
                try:
                    _fref = a.get("file") or a.get("file_ref") or "last"   # paper Table 5: upload(id, file)
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

# ---------------------------------------------------------------------------------------------------
# PB uses the PB-native parity adapter (PbNativeEnv). It wraps the UPSTREAM PhysicianBench tool registry
# (agent/tool_registry.py + tools/fhir_api_functions.py + tools/file_tools.py) so the agent gets the
# FAITHFUL tool surface: full JSON schemas (code/date/count/page_limit filters), real Bundle.link[next]
# pagination, and structured create helpers. It is the only env registered for "fhir".
ENV_REGISTRY = {"fhir": PbNativeEnv, "gui": GuiEnvReal, "tool_sandbox": ToolSandboxEnv}

def make_env(task, **kw):
    et = (task.get("environment") or {}).get("type")
    if et == "gui":
        # ONE GUI implementation: the real Playwright/Next.js portal; there is no in-memory mock.
        # A GUI task must run on a node that can launch chromium (browser
        # deps + CPU/memory; NO GPU -- HAB is DOM-based, agent+judge are remote APIs). Without a
        # launchable chromium this raises at reset() rather than silently running a fake portal.
        return GuiEnvReal(task, **kw)
    return ENV_REGISTRY[et](task, **kw)
