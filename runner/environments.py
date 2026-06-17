"""Environment adapters — dispatch by unified task `environment.type`.

base: reset() / available_tools() / call_tool(name,args) / teardown().
FhirEnv is real (hits the HAPI FHIR server). GuiEnv / ToolSandboxEnv are skeleton stubs
(to be filled with Playwright / agentlego). Registry maps environment.type -> adapter.
"""
import os, json, urllib.request, urllib.parse

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
        return json.load(urllib.request.urlopen(req, timeout=30))
    def call_tool(self, name, args):
        if name == "fhir_search":
            rt = args.get("resourceType", ""); params = {k: v for k, v in args.items() if k != "resourceType"}
            return self._get(f"/{rt}?" + urllib.parse.urlencode(params))
        if name == "fhir_read":
            return self._get(f"/{args['resourceType']}/{args['id']}")
        if name == "fhir_create":
            res = args.get("resource", args); rt = res.get("resourceType", "")
            data = json.dumps(res).encode()
            req = urllib.request.Request(f"{self.base}/{rt}", data=data, method="POST",
                                         headers={"Content-Type": "application/fhir+json"})
            return json.load(urllib.request.urlopen(req, timeout=30))
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
            for pre in ("workspace/output/", "workspace/"):
                if rel.startswith(pre):
                    rel = rel[len(pre):]
                    break
            fp = os.path.join(self.workspace, rel)
            os.makedirs(os.path.dirname(fp) or self.workspace, exist_ok=True)
            open(fp, "w").write(args.get("content", "")); return {"written": fp}
        return {"error": f"unknown tool {name}"}

class GuiEnv(EnvironmentAdapter):
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
    """MedCTA v0: REPLAY env — returns the cached tool outputs recorded in the reference trace (pi),
    so a ReplayAgent can reproduce the gold trajectory and validate the ToolAcc/ArgAcc/Gacc scorer
    paths WITHOUT a VLM/GPU/API. v1 = real agentlego 5 tools + frozen GoogleSearch + VLM backend.
    """
    type = "tool_sandbox"
    def __init__(self, task, **kw):
        super().__init__(task)
        # cached tool outputs from reference_trace (role==tool messages), in order
        trace = (task.get("reference") or {}).get("reference_trace") or []
        self._cached = [ev.get("content") for ev in trace if ev.get("role") == "tool"]
        self._i = 0
    def reset(self):
        self._i = 0
    def call_tool(self, name, args):
        out = self._cached[self._i] if self._i < len(self._cached) else {"replayed": True}
        self._i += 1
        return {"tool": name, "args": args, "output": out, "mode": "replay"}

ENV_REGISTRY = {"fhir": FhirEnv, "gui": GuiEnv, "tool_sandbox": ToolSandboxEnv}
def make_env(task, **kw):
    et = (task.get("environment") or {}).get("type")
    return ENV_REGISTRY[et](task, **kw)
