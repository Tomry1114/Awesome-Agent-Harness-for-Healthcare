# Ensure BOTH runner/ (flat top-level modules: run, substrate, environments, harness, ...) and
# runner/scorers/ (the scoring machinery: scoring, aggregate_report, rescore_judges, score, dim_*,
# lifecycle_exec, proxy_verifiers, governance, governance_contract)
# are importable by bare name during test collection, regardless of cwd. pytest auto-loads this.
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (os.path.join(_HERE, "scorers"), _HERE):
    if _d not in sys.path:
        sys.path.insert(0, _d)
