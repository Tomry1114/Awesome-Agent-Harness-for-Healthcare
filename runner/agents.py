"""Agents — produce actions given (goal, context, available_tools, last_observation, last_result).

AgentBase.act(state) -> {"type":"tool_call","tool":...,"args":...} | {"type":"final","answer":...}

StubAgent is a DETERMINISTIC REGRESSION-TEST agent for harness plumbing — NOT a clinical baseline.
It does not read the full goal, does not decide whether prescribing is appropriate, does not check
renal/labs, and only demonstrates the governance pipeline. A real benchmark baseline = an LLM agent
using the upstream semantic FHIR tools (B-line, needs API key).
It IS reactive on allergies: after querying AllergyIntolerance it picks a pool drug that does not
match any of the patient's allergens (so it stays safe even as synthetic allergens vary).
"""
STUB_TAG = {"system": "http://medical-harness/tags", "code": "stub-run", "display": "stub agent created"}

class AgentBase:
    name = "base"
    def act(self, state): raise NotImplementedError

class StubAgent(AgentBase):
    name = "stub"
    SAFE_POOL = ["fexofenadine 180 mg oral tablet", "loratadine 10 mg oral tablet",
                 "cetirizine 10 mg oral tablet", "levocetirizine 5 mg oral tablet",
                 "desloratadine 5 mg oral tablet", "acetaminophen 500 mg oral tablet"]
    def __init__(self, task):
        import re
        self.task = task
        self.mrn = (task.get("context", {}) or {}).get("patient_ref")
        self.is_fhir = (task.get("environment") or {}).get("type") == "fhir" and self.mrn
        self.phase = 0
        self.allergens = []
        # deliverable file the instruction asks for (so native_pytest read_output_file finds it)
        m = re.search(r"workspace/output/([\w.\-]+)", task.get("goal", "") or "")
        self.deliverable = m.group(1) if m else None
    def _pick_safe_med(self):
        bad = " ".join(self.allergens).lower()
        for m in self.SAFE_POOL:
            ing = m.split()[0].lower()
            if ing not in bad:
                return m
        return self.SAFE_POOL[-1]  # acetaminophen fallback
    def act(self, state):
        if not self.is_fhir:
            return {"type": "final", "answer": "stub: environment not yet supported"}
        # capture allergens from the allergy-query result
        lr = state.get("last_result")
        if self.phase == 1 and isinstance(lr, dict):
            for e in lr.get("entry", []) or []:
                txt = ((e.get("resource", {}) or {}).get("code", {}) or {}).get("text")
                if txt: self.allergens.append(txt)
        p = self.phase; self.phase += 1
        if p == 0:
            return {"type": "tool_call", "tool": "fhir_search",
                    "args": {"resourceType": "AllergyIntolerance", "patient": f"Patient/{self.mrn}"}}
        if p == 1:
            return {"type": "tool_call", "tool": "fhir_search",
                    "args": {"resourceType": "MedicationRequest", "patient": f"Patient/{self.mrn}", "_summary": "count"}}
        if p == 2:
            self._med = self._pick_safe_med()
            return {"type": "tool_call", "tool": "fhir_create", "args": {"resource": {
                "resourceType": "MedicationRequest", "status": "active", "intent": "order",
                "medicationCodeableConcept": {"text": self._med}, "subject": {"reference": f"Patient/{self.mrn}"},
                "meta": {"tag": [STUB_TAG]}}}}
        if p == 3 and self.deliverable:
            note = (f"# Management Plan (stub)\nReviewed allergies: {self.allergens or '[none]'}.\n"
                    f"Prescribed {getattr(self, '_med', 'n/a')} (selected to avoid documented allergies).\n"
                    f"NOTE: stub regression-test output, not clinical content.\n")
            return {"type": "tool_call", "tool": "write_file", "args": {"path": self.deliverable, "content": note}}
        return {"type": "final", "answer": f"Reviewed allergies {self.allergens or '[none]'}; prescribed a non-conflicting medication; wrote {self.deliverable or 'no deliverable'}."}

class StubGuiAgent(AgentBase):
    """v0 GUI regression agent: generic portal workflow (navigate→view→document→submit). It does NOT
    know the gold action targets (hidden in checkpoints), so most JMESPath checkpoints will fail —
    that's expected; the point is the GUI substrate path executes. NOT a baseline."""
    name = "stub"
    def __init__(self, task):
        self.url = ((task.get("environment") or {}).get("config", {}).get("website") or {}).get("url")
        self.script = [
            {"type": "tool_call", "tool": "navigate", "args": {"url": self.url}},
            {"type": "tool_call", "tool": "click", "args": {"target": "viewDetails"}},
            {"type": "tool_call", "tool": "type", "args": {"field": "triageNote", "text": "Reviewed available information and documented disposition (stub)."}},
            {"type": "tool_call", "tool": "submit", "args": {"target": "decision"}},
            {"type": "final", "answer": "stub GUI workflow complete"},
        ]
        self._i = 0
    def act(self, state):
        a = self.script[self._i] if self._i < len(self.script) else {"type": "final", "answer": "done"}
        self._i += 1
        return a

class ReplayAgent(AgentBase):
    """GOLD-REPLAY validation agent (MedCTA v0): reproduces the reference trajectory (pi) from
    task.reference — emits its tool_calls then the gold final answer. Used ONLY to validate the
    scorer paths (ToolAcc/ArgAcc/Gacc return PASS on gold behavior). NOT a real agent: a real agent
    never sees task.reference."""
    name = "replay"
    def __init__(self, task):
        trace = (task.get("reference") or {}).get("reference_trace") or []
        self.steps = []
        for ev in trace:
            if ev.get("role") == "assistant" and ev.get("tool_calls"):
                for tc in ev["tool_calls"]:
                    fn = tc.get("function", {})
                    self.steps.append({"type": "tool_call", "tool": fn.get("name"), "args": fn.get("arguments") or {}})
            elif ev.get("role") == "assistant" and ev.get("content"):
                self.steps.append({"type": "final", "answer": ev["content"]})
        if not self.steps or self.steps[-1]["type"] != "final":
            ga = ((task.get("reference") or {}).get("gold_answer") or {}).get("whitelist") or [[""]]
            self.steps.append({"type": "final", "answer": ga[0][0] if ga and ga[0] else ""})
        self._i = 0
    def act(self, state):
        a = self.steps[self._i] if self._i < len(self.steps) else {"type": "final", "answer": "done"}
        self._i += 1
        return a

class ScriptedAgent(AgentBase):
    """GOLD-PATH validation agent: replays a fixed action list from env MH_SCRIPT (JSON list of
    {"tool":..., "args":...}; appends a final). Proves a task has a reachable ground truth and the
    scorer returns success for a correct trajectory. NOT a real agent."""
    name = "scripted"
    def __init__(self, task):
        import json, os
        self.steps = json.loads(os.environ.get("MH_SCRIPT", "[]"))
        for st in self.steps:
            st.setdefault("type", "tool_call")
        if not self.steps or self.steps[-1].get("type") != "final":
            self.steps.append({"type": "final", "answer": "scripted gold path complete"})
        self._i = 0
    def act(self, state):
        a = self.steps[self._i] if self._i < len(self.steps) else {"type": "final", "answer": "done"}
        self._i += 1
        return a

def make_agent(name, task):
    et = (task.get("environment") or {}).get("type")
    if name == "scripted":
        return ScriptedAgent(task)
    if name in ("gpt5", "openai"):
        from openai_agent import OpenAIToolAgent
        return OpenAIToolAgent(task)
    if name == "qwen":
        from qwen_agent import QwenToolAgent
        return QwenToolAgent(task)
    if name == "stub" and et == "gui":
        return StubGuiAgent(task)
    if name == "stub" and et == "tool_sandbox":
        return ReplayAgent(task)   # MedCTA stub == gold replay (no VLM/key)
    return AGENT_REGISTRY[name](task)

AGENT_REGISTRY = {"stub": StubAgent, "replay": ReplayAgent, "scripted": ScriptedAgent}
