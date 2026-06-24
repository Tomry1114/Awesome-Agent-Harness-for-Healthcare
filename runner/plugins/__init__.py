"""runner/plugins -- the BenchmarkPlugin package.

Importing this package auto-registers every benchmark plugin into the substrate registry (each sibling
module calls substrate.register_plugin(...) at import time). substrate.py imports this package at the very
END of its module body (after all core + shared-helper definitions exist), so `import substrate` alone
yields list_plugins() == the three registered benchmarks. The dependency is strictly one-directional:

    substrate (core, no benchmark literal)  <--imported by--  plugins.<name>  (benchmark-specific)
    substrate  --imports at end-->  plugins   (only to trigger registration; no name leaks back into core)

A 4TH DATASET = drop runner/plugins/<name>.py (register a PLUGIN dict via substrate.register_plugin) and add
its entry to spec/registry.json, then list it below. No edit to substrate core or the dimension evaluators."""
from . import medcta            # noqa: F401  (import side effect: register_plugin)
from . import physicianbench    # noqa: F401
from . import healthadminbench  # noqa: F401

__all__ = ["medcta", "physicianbench", "healthadminbench"]
