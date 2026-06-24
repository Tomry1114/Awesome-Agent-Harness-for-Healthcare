"""runner/plugins -- the BenchmarkPlugin package (AUTO-DISCOVERING).

Importing this package auto-registers every benchmark plugin into the substrate registry. Each sibling
module calls substrate.register_plugin(...) at import time; this __init__ DISCOVERS those modules instead of
hardcoding their names, so a 4th dataset is truly drop-a-file: dropping runner/plugins/<name>.py (which
registers a PLUGIN dict) makes it load with NO edit here.

Discovery = pkgutil.iter_modules over this package's __path__, importlib.import_module each non-underscore
module under the package (package-relative, "%s.%s" % (__name__, name)) so the import path is identical to
the old `from . import <name>` and no second copy of the module is created (no duplicate registry entry).

substrate.py imports this package at the very END of its module body (after all core + shared-helper
definitions exist), so `import substrate` alone yields list_plugins() == every registered benchmark. The
dependency stays strictly one-directional:

    substrate (core, no benchmark literal)  <--imported by--  plugins.<name>  (benchmark-specific)
    substrate  --imports at end-->  plugins   (only to trigger registration; no name leaks back into core)
"""
import importlib as _importlib
import pkgutil as _pkgutil

# Discover & import every non-underscore sibling module exactly once (package-relative). iter_modules over
# __path__ is deterministic per filesystem; importlib.import_module is idempotent (sys.modules cache) so a
# re-import of this package never double-registers. We sort for a stable, reproducible load order.
__all__ = []
for _mod in sorted(m.name for m in _pkgutil.iter_modules(__path__)):
    if _mod.startswith("_"):
        continue
    _importlib.import_module("%s.%s" % (__name__, _mod))
    __all__.append(_mod)

del _importlib, _pkgutil, _mod
