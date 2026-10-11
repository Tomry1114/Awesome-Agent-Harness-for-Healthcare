"""Conformance test: the unified adapter registry resolves the 3 datasets (by env_type and by bench name) and
every adapter carries its identity and manifest. Run: from runner/ with PYTHONPATH=.  ->  python3 adapters/test_registry.py"""
import sys
import adapters as A

R = []
def ck(name, cond):
    R.append((name, bool(cond)))

# resolve by env_type
ck("fhir->PB",         type(A.get_adapter(env_type="fhir")).__name__ == "PBAdapter")
ck("gui->HAB",         type(A.get_adapter(env_type="gui")).__name__ == "HABAdapter")
ck("tool_sandbox->MedCTA", type(A.get_adapter(env_type="tool_sandbox")).__name__ == "MedctaAdapter")
# resolve by bench name (wins over env_type)
ck("bench PhysicianBench",  A.get_adapter(bench="PhysicianBench").name == "PhysicianBench")
ck("bench HealthAdminBench", A.get_adapter(bench="HealthAdminBench").name == "HealthAdminBench")
ck("bench MedCTA",          A.get_adapter(bench="MedCTA").name == "MedCTA")
ck("bench wins over env",   A.get_adapter(env_type="fhir", bench="MedCTA").name == "MedCTA")
# unknown -> None
ck("unknown env -> None",   A.get_adapter(env_type="nope") is None)
ck("no args -> None",       A.get_adapter() is None)
# every adapter carries identity + manifest
for a in A.all_adapters():
    ck("%s env_type set" % a.name, a.env_type in ("fhir", "gui", "tool_sandbox"))
    ck("%s manifest set" % a.name, isinstance(a.manifest_yaml, str) and a.manifest_yaml.endswith(".yaml"))
# applies(gate) must be boolean-identical to an `env_type == "gui"` check for every GUI gate, across every
# env_type (including None and an unknown value), so run.py can ask the adapter instead of testing env_type.
GUI_GATES = ["obligation_gate", "collection_nudge"]
for env in ["gui", "fhir", "tool_sandbox", None, "bogus"]:
    a = A.get_adapter(env_type=env) or A.HarnessAdapter()
    for g in GUI_GATES:
        ck("applies(%s)@%s == (env==gui)" % (g, env), a.applies(g) == (env == "gui"))
# a non-gui gate name is never claimed by anyone
ck("unknown gate never applies", not A.get_adapter(env_type="gui").applies("no_such_gate"))

ok = sum(1 for _, c in R if c)
for n, c in R:
    if not c:
        print("FAIL", n)
print("\n%d/%d passed" % (ok, len(R)))
sys.exit(0 if ok == len(R) else 1)
