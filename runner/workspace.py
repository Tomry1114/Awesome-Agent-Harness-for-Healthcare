# -*- coding: utf-8 -*-
"""Harness workspaces: the file-based part of a harness, one directory per workspace.

The fixed shell (this runner) owns the environment, the safety kernel, the scorer, the model transport and the
budget. A workspace owns what may be changed without touching the shell: the agent program and the components it reads
(system prompt, prompt fragments, tool descriptions), the middleware the shell calls (runner/middleware_host.py), the
kernel middleware the harness kernel runs after its fixed capabilities (runner/harness/__init__.py), the policy
modules composed after the shell's clinical modules, the completion-obligation content the shell's obligation
engine composes (runner/harness/obligation_gate.py) and the allocation of the shell's total semantic budget (manifest
semantic_budget, runner/harness/semantic_budget.py). Which workspace a run uses is part of the run's input, like the
model.

Identity: a workspace is identified by workspace_digest over its files; a run resolves its workspace once
(runner/run.py resolve_workspace), refuses one whose manifest does not declare the run's benchmark and env_type, and
hands the path to every reader. No reader resolves a default workspace of its own.
"""
import ast
import copy
import hashlib
import importlib.util
import json
import os
import random
import re
import sys

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The default (H0) workspace per substrate (workspaces/<bench key>/h0). The one runtime reader is runner/run.py
# resolve_workspace: a run that names no workspace runs its substrate's H0. Everything else receives the run's workspace.
H0 = {"fhir": os.path.join(REPO, "workspaces", "pb", "h0"),
      "gui": os.path.join(REPO, "workspaces", "hab", "h0"),
      "tool_sandbox": os.path.join(REPO, "workspaces", "medcta", "h0")}


# The model protocols an agent program is written for. A workspace names one program (and its system prompt) per
# protocol: `agent_program` / `system_prompt` for function calling, `agent_program_text` / `system_prompt_text` for
# the <tool_call>/<answer> text protocol. The protocol is the model's (runner/models.yaml); WorkspaceAgent runs the
# program for its model's protocol.
from models import FUNCTION_CALLING, TEXT, PROTOCOLS  # noqa: E402

# The environment config an agent program may read. The upstream HealthAdminBench harness hands the payer portal and
# the challenge type to its agent prompt (get_hints_for_task); the rest of the config (the native task config, whether
# the task is possible) and every other hidden task field (checkpoints, reference, expected_outcome, policy, scoring)
# never reaches a program.
PUBLIC_ENVIRONMENT_CONFIG = ("payer_portal", "challenge_type")


class WorkspaceError(RuntimeError):
    """A workspace that cannot be loaded as declared. Raised before the run starts, never mid-episode."""


# ------------------------------------------------------------------------------------------------------------ identity
# The digest covers every regular file under the workspace directory, whatever its kind,
# except Python's byte-code cache. A symlink anywhere under it, or a file resolving outside it, is refused: the digest
# would not name what runs.
DIGEST_EXCLUDED_DIRS = ("__pycache__",)
DIGEST_EXCLUDED_SUFFIXES = (".pyc",)
CANDIDATE_FILE = "candidate.json"     # optional, in the directory that contains the workspace directory; declares parent_digest
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _inside(path, real_root):
    """Does `path`, with every symlink resolved, lie at or below `real_root` (itself a realpath)?"""
    real = os.path.realpath(path)
    return real == real_root or real.startswith(real_root + os.sep)


def _digest_files(path):
    """[(relative path with "/" separators, absolute path)] of every file the digest covers, sorted by relative path.
    Refuses (WorkspaceError) a workspace that is not a directory or is itself a symlink, a symlink anywhere under it
    (file or directory, excluded names included), an entry that is not a regular file, a file resolving outside, and a
    directory it cannot list (os.walk would otherwise leave it out without a word)."""
    top = os.path.abspath(path)
    if os.path.islink(top) or not os.path.isdir(top):
        raise WorkspaceError("workspace %s is not a directory (a symlink is refused)" % top)
    real_root = os.path.realpath(top)
    out = []

    def unreadable(err):
        raise WorkspaceError("workspace %s: cannot list %s: %s" % (top, getattr(err, "filename", None), err))
    for d, dirs, files in os.walk(top, onerror=unreadable):
        for name in dirs + files:
            full = os.path.join(d, name)
            if os.path.islink(full):
                raise WorkspaceError("workspace %s: %s is a symlink; a workspace holds regular files only"
                                     % (top, os.path.relpath(full, top)))
        dirs[:] = sorted(n for n in dirs if n not in DIGEST_EXCLUDED_DIRS)
        for name in files:
            if name.endswith(DIGEST_EXCLUDED_SUFFIXES):
                continue
            full = os.path.join(d, name)
            rel = os.path.relpath(full, top)
            if not os.path.isfile(full):
                raise WorkspaceError("workspace %s: %s is not a regular file" % (top, rel))
            if not _inside(full, real_root):
                raise WorkspaceError("workspace %s: %s resolves outside the workspace" % (top, rel))
            out.append((rel.replace(os.sep, "/"), full))
    return sorted(out)


def component_digests(path):
    """{relative path: sha256 hex of the file's bytes} for every file workspace_digest covers."""
    out = {}
    for rel, full in _digest_files(path):
        with open(full, "rb") as f:
            out[rel] = hashlib.sha256(f.read()).hexdigest()
    return out


def digest_of(components):
    """The workspace digest of a component_digests map: "sha256:" + the sha256 over its (relative path, sha256) pairs in
    sorted path order, each pair fed as the UTF-8 path, one NUL byte, the hex digest and one newline."""
    h = hashlib.sha256()
    for rel in sorted(components):
        h.update(rel.encode("utf-8") + b"\0" + components[rel].encode("ascii") + b"\n")
    return "sha256:" + h.hexdigest()


def workspace_digest(path):
    """"sha256:<hex>" identifying the workspace directory `path` by the bytes of every file under it (component_digests,
    digest_of). Two directories with the same relative paths and bytes have the same digest wherever they are."""
    return digest_of(component_digests(path))


def parent_digest(path):
    """The parent_digest declared by the candidate.json in the directory that contains the workspace directory
    (<dir>/candidate.json beside <dir>/workspace/), None when there is no such file. A candidate.json that is not a
    JSON object, or whose parent_digest is neither null nor "sha256:<64 hex>", is refused."""
    p = os.path.join(os.path.dirname(os.path.abspath(path)), CANDIDATE_FILE)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            doc = json.load(f)
    except ValueError as e:
        raise WorkspaceError("%s: not JSON: %s" % (p, e))
    pd = doc.get("parent_digest") if isinstance(doc, dict) else None
    if not isinstance(doc, dict) or not (pd is None or (isinstance(pd, str) and _DIGEST_RE.match(pd))):
        raise WorkspaceError("%s: parent_digest must be null or sha256:<64 hex>, got %r" % (p, pd))
    return pd


def identity(path, benchmark, env_type):
    """What a run records of the workspace it runs (result.json `workspace`): {path (repository-relative when inside the
    repository), digest, component_digests, parent_digest}. Refuses (WorkspaceError) a workspace the digest refuses and
    one whose manifest.yaml does not declare `benchmark` and `env_type` equal to the run's."""
    comps = component_digests(path)
    ws = Workspace(path)
    wrong = ["manifest %s is %r, the run's is %r" % (key, ws.manifest.get(key), want)
             for key, want in (("benchmark", benchmark), ("env_type", env_type)) if ws.manifest.get(key) != want]
    if wrong:
        raise WorkspaceError("REFUSING workspace %s: %s" % (ws.path, "; ".join(wrong)))
    return {"path": repo_path(ws.path), "digest": digest_of(comps), "component_digests": comps,
            "parent_digest": parent_digest(ws.path)}


def public_task(task):
    """What workspace code is given of a task (agent programs, host middleware, the modules they load): the goal, the
    context, the tools and the public environment view, as a copy (nothing a program changes reaches the shell's task).
    Checkpoints, reference, expected outcome, scoring, policy, native_config, task_type and every other field stay in
    the shell."""
    task = task if isinstance(task, dict) else {}
    env = task.get("environment") or {}
    config = env.get("config") or {}
    return copy.deepcopy({"goal": task.get("goal"), "context": task.get("context"),
                          "available_tools": task.get("available_tools") or [],
                          "environment": {"type": env.get("type"),
                                          "config": {k: config[k] for k in PUBLIC_ENVIRONMENT_CONFIG if k in config}}})


# ------------------------------------------------------------------------------------------------------------ isolation
# Workspace code imports from the runner only runner/workspace_api.py (its __all__), and
# from the standard library only the modules below, each through a read-only facade naming the attributes it offers
# (None: every public attribute that is not itself a module). Two enforcements, one allow-list:
#   load time   _module runs every workspace file with WORKSPACE_BUILTINS: its __import__ is _guarded_import, and the
#               builtins that reach code, files, frames or namespaces (eval, exec, compile, open, globals, locals, vars,
#               breakpoint, input, ...) are absent. The builtins belong to the module's globals, so the guard covers
#               every import the module's code runs, at load and inside a function called later.
#   static      Workspace.check -> isolation_problems: the same import allow-list, no relative import, none of the
#               refused builtins, no dunder name or attribute beyond ALLOWED_DUNDERS (the introspection route to
#               closures, globals, classes and subclasses), none of the frame / traceback attributes in
#               REFUSED_ATTRIBUTES, and getattr / setattr / delattr / hasattr only called directly with a literal
#               attribute name (an alias or a computed name would pass a dunder the parser cannot see).
# This is not a sandbox against code written to escape; it removes every direct path (an import, a raw env, a writable
# ledger, a raw judge) and the known introspection paths, so reaching the shell is a refused act, not a one-line import.
WORKSPACE_API = "workspace_api"
STDLIB_IMPORTS = {
    "re": None,
    "json": ("dumps", "loads", "JSONDecodeError"),
    "hashlib": ("md5", "sha1", "sha256"),
    "typing": ("Any", "Callable", "Dict", "Iterable", "List", "Mapping", "Optional", "Sequence", "Set", "Tuple",
               "Union"),
    "dataclasses": ("dataclass", "field", "fields", "asdict", "replace"),
    "collections": ("Counter", "OrderedDict", "defaultdict", "deque", "namedtuple"),
    "copy": ("copy", "deepcopy"),
    "math": None,
    "itertools": None,
    "heapq": None,
    "bisect": None,
    "posixpath": ("basename", "dirname", "join", "normpath", "split", "splitext"),   # path strings, no file access
}
REFUSED_BUILTINS = ("__import__", "eval", "exec", "compile", "open", "globals", "locals", "vars", "breakpoint", "input",
                    "help", "exit", "quit", "copyright", "credits", "license")
ALLOWED_DUNDERS = frozenset(("__name__", "__qualname__", "__doc__", "__file__"))
REFUSED_ATTRIBUTES = frozenset(("gi_frame", "gi_code", "cr_frame", "cr_code", "ag_frame", "ag_code", "f_back",
                                "f_globals", "f_locals", "f_builtins", "f_code", "tb_frame", "tb_next"))
_ATTRIBUTE_BUILTINS = ("getattr", "setattr", "delattr", "hasattr")


class WorkspaceImportError(ImportError):
    """A workspace module imported something outside the allow-list."""


class _Namespace(type):
    """The metaclass of a read-only facade: attributes are fixed at creation; setting or deleting one is refused and
    a missing one names the allow-list."""

    def __setattr__(cls, name, value):
        raise AttributeError("%s is read-only to workspace code (tried to set %s)" % (cls.__name__, name))

    def __delattr__(cls, name):
        raise AttributeError("%s is read-only to workspace code (tried to delete %s)" % (cls.__name__, name))

    def __getattr__(cls, name):
        raise AttributeError("%s offers workspace code no %r (runner/workspace.py STDLIB_IMPORTS, "
                             "runner/workspace_api.py __all__)" % (cls.__name__, name))


def namespace(name, attributes):
    """A read-only facade named `name` with exactly `attributes` ({name: value}). Values are read as class attributes,
    so a function stays a plain function."""
    return _Namespace(name, (object,), dict(attributes, __slots__=()))


_FACADES = {}


def _facade(name):
    """The facade a workspace import of `name` yields: workspace_api's __all__, or the allow-listed attributes of a
    standard-library module (never an attribute that is a module)."""
    if name not in _FACADES:
        import importlib
        import types
        mod = importlib.import_module(name)
        if name == WORKSPACE_API:
            names = mod.__all__
        else:
            names = STDLIB_IMPORTS[name]
            if names is None:
                names = [n for n in dir(mod) if not n.startswith("_")]
        attrs = {n: getattr(mod, n) for n in names if not isinstance(getattr(mod, n), types.ModuleType)}
        _FACADES[name] = namespace(name, attrs)
    return _FACADES[name]


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    """__import__ as workspace code has it: workspace_api or an allow-listed standard-library module, as a facade."""
    if level:
        raise WorkspaceImportError("workspace code has no relative import (%s); load workspace modules with "
                                   "workspace_api.load_relative" % (name,))
    if name == WORKSPACE_API or name in STDLIB_IMPORTS:
        return _facade(name)
    raise WorkspaceImportError("workspace code may import only %s and the standard-library modules %s; %r is refused"
                               % (WORKSPACE_API, ", ".join(sorted(STDLIB_IMPORTS)),
                                                                   name))


def _workspace_builtins():
    import builtins
    out = {k: v for k, v in vars(builtins).items() if k not in REFUSED_BUILTINS}
    out["__import__"] = _guarded_import
    return out


WORKSPACE_BUILTINS = _workspace_builtins()


def _dunder(name):
    return name.startswith("__") and name.endswith("__")


def _import_problems(node):
    """Problems of one Import / ImportFrom node against the allow-list."""
    out = []
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name != WORKSPACE_API and alias.name not in STDLIB_IMPORTS:
                out.append("import %s is refused" % alias.name)
        return out
    if node.level:
        return ["relative import from %s is refused" % ("." * node.level + (node.module or ""))]
    if node.module == WORKSPACE_API:
        import workspace_api
        offered = set(workspace_api.__all__)
    elif node.module in STDLIB_IMPORTS:
        offered = STDLIB_IMPORTS[node.module]
        offered = None if offered is None else set(offered)
    else:
        return ["from %s import is refused" % node.module]
    for alias in node.names:
        if alias.name == "*" or (offered is not None and alias.name not in offered) or \
                (offered is None and alias.name.startswith("_")):
            out.append("from %s import %s is refused" % (node.module, alias.name))
    return out


def source_isolation_problems(source, where):
    """The isolation problems of one workspace Python source (see the isolation note above)."""
    try:
        tree = ast.parse(source, filename=where)
    except SyntaxError as e:
        return ["%s: does not parse: %s" % (where, e)]
    out = []
    direct = set()           # Name nodes that are the callee of a well-formed getattr/setattr/delattr/hasattr call
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _ATTRIBUTE_BUILTINS:
            arg = node.args[1] if len(node.args) > 1 else None
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and \
                    (not _dunder(arg.value) or arg.value in ALLOWED_DUNDERS) and arg.value not in REFUSED_ATTRIBUTES:
                direct.add(id(node.func))
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            out += ["%s:%d: %s" % (where, line, p) for p in _import_problems(node)]
        elif isinstance(node, ast.Name):
            if node.id in REFUSED_BUILTINS or node.id in ("__builtins__", "__loader__", "__spec__") or \
                    (_dunder(node.id) and node.id not in ALLOWED_DUNDERS):
                out.append("%s:%d: the name %s is refused" % (where, line, node.id))
            elif node.id in _ATTRIBUTE_BUILTINS and id(node) not in direct:
                out.append("%s:%d: %s is allowed only as a direct call with a literal attribute name" %
                           (where, line, node.id))
        elif isinstance(node, ast.Attribute):
            if (_dunder(node.attr) and node.attr not in ALLOWED_DUNDERS) or node.attr in REFUSED_ATTRIBUTES:
                out.append("%s:%d: the attribute %s is refused" % (where, line, node.attr))
    return out


def isolation_problems(path):
    """Every isolation problem of the Python files of the workspace directory `path` (the files its digest covers)."""
    out = []
    for rel, full in _digest_files(path):
        if rel.endswith(".py"):
            with open(full, encoding="utf-8") as f:
                out += source_isolation_problems(f.read(), rel)
    return out


_MODULE_PREFIX = "_workspace_"


def repo_path(path):
    """A path relative to the repository when it is inside it, else absolute."""
    path = os.path.abspath(path)
    return os.path.relpath(path, REPO) if path.startswith(REPO + os.sep) else path


def component_path(obj):
    """The absolute path of the workspace file an object (or class) was loaded from; None for shell code."""
    cls = obj if isinstance(obj, type) else type(obj)
    mod = sys.modules.get(cls.__module__)
    if mod is None or not cls.__module__.startswith(_MODULE_PREFIX):
        return None
    return os.path.abspath(mod.__file__)


def _module(path):
    """A workspace file as a module, executed once per process the way an import is, so module-level state behaves
    as it does for an imported module. Keyed by absolute path: two workspace directories never share a module."""
    path = os.path.abspath(path)
    stem = re.sub(r"\W", "_", os.path.splitext(os.path.basename(path))[0])
    name = "%s%s_%s" % (_MODULE_PREFIX, hashlib.sha1(path.encode("utf-8")).hexdigest()[:12], stem)
    if name in sys.modules:
        return sys.modules[name]
    if not os.path.isfile(path):
        raise WorkspaceError("workspace file %s missing" % path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # Isolation: the module's code runs with the workspace builtins (the guarded __import__, no
    # eval/exec/open/...), and without its spec and loader (a loader reads any file it is asked for).
    mod.__dict__["__builtins__"] = WORKSPACE_BUILTINS
    mod.__spec__ = None
    mod.__loader__ = None
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        del sys.modules[name]
        raise
    return mod


# Workspace code imports workspace code only through load_relative. Per absolute path of the loading file, the absolute
# paths of the modules it loaded, in load order; instrumentation lists them beside the component that loaded them.
_LOADED_BY = {}


def workspace_of(path):
    """The workspace directory a workspace file belongs to: the nearest directory above `path` holding a manifest.yaml,
    or None. A workspace module that hands its own workspace's content to a shell reader (e.g. the obligation engine)
    passes workspace_of(__file__): the directory the run loaded it from."""
    return _workspace_root(path)


def _workspace_root(path):
    """The nearest directory above `path` holding a manifest.yaml (the workspace `path` belongs to), or None."""
    d = os.path.dirname(os.path.abspath(path))
    while True:
        if os.path.isfile(os.path.join(d, "manifest.yaml")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def load_relative(by, rel):
    """A module of the same workspace, loaded as the shell loads a component (_module: executed once per process,
    keyed by absolute path, so two workspace directories never share a module). `by` is the __file__ of the workspace
    file asking, `rel` a path relative to its directory. No workspace directory is ever put on sys.path; this is the only
    way workspace code imports workspace code. The target must lie inside by's workspace once every symlink is resolved
    (realpath: a "..", an absolute path or a symlink out of the workspace is refused). The dependency is recorded, so
    instrumentation lists every module a loaded component runs (modules_loaded_by)."""
    by = os.path.abspath(by)
    path = os.path.abspath(os.path.join(os.path.dirname(by), rel))
    root = _workspace_root(by)
    if root is None or not _inside(path, os.path.realpath(root)) or os.path.realpath(path) == os.path.realpath(root):
        raise WorkspaceError("%s: %s is not a module of its workspace" % (by, rel))
    deps = _LOADED_BY.setdefault(by, [])
    if path not in deps:
        deps.append(path)
    return _module(path)


def modules_loaded_by(path):
    """Every workspace module the file `path` loaded through load_relative, transitively, in first-load order."""
    out, seen = [], set([os.path.abspath(path)])

    def walk(p):
        for dep in _LOADED_BY.get(p, ()):
            if dep not in seen:
                seen.add(dep)
                out.append(dep)
                walk(dep)
    walk(os.path.abspath(path))
    return out


class Workspace(object):
    def __init__(self, path, protocol=FUNCTION_CALLING):
        self.path = os.path.abspath(path)
        if protocol not in PROTOCOLS:
            raise WorkspaceError("unknown protocol %r (one of %s)" % (protocol, ", ".join(PROTOCOLS)))
        self.protocol = protocol
        mp = os.path.join(self.path, "manifest.yaml")
        if not os.path.isfile(mp):
            raise WorkspaceError("no manifest.yaml in %s" % self.path)
        self.manifest = yaml.safe_load(open(mp, encoding="utf-8")) or {}
        if not isinstance(self.manifest, dict):
            raise WorkspaceError("%s: manifest.yaml is not a mapping" % self.path)
        self._descriptions = {}
        self._fragments = {}

    def _path(self, rel):
        """This workspace's path `rel`. One resolving outside the workspace (a "..", an absolute path or a symlink out
        of it) is refused: whatever a manifest names is part of the workspace the digest identifies."""
        p = os.path.join(self.path, rel)
        if not _inside(p, os.path.realpath(self.path)):
            raise WorkspaceError("%s: %s resolves outside the workspace" % (self.path, rel))
        return p

    def key(self, component):
        """The manifest key of a per-protocol component (agent_program, system_prompt) for this workspace's protocol."""
        return component if self.protocol == FUNCTION_CALLING else "%s_%s" % (component, self.protocol)

    def _file(self, key):
        if not self.manifest.get(key):
            raise WorkspaceError("%s: manifest.yaml lacks %r" % (self.path, key))
        return self._path(self.manifest[key])

    def load(self, rel, cls):
        """The class `cls` defined in this workspace's file `rel`."""
        mod = _module(self._path(rel))
        if not hasattr(mod, cls):
            raise WorkspaceError("%s defines no %s" % (os.path.join(self.path, rel), cls))
        return getattr(mod, cls)

    @property
    def system_prompt(self):
        """The protocol's system prompt file, without the one trailing newline a text file ends with."""
        return _read_text(self._file(self.key("system_prompt")))

    def prompt_fragment(self, name):
        """prompt_fragments/<name>.md without its trailing newline: a piece of prompt text the agent program
        assembles (e.g. per-task guidance)."""
        if name not in self._fragments:
            p = os.path.join(self._file("prompt_fragments"), "%s.md" % name)
            if not _inside(p, os.path.realpath(self.path)):
                raise WorkspaceError("%s: prompt fragment %r resolves outside the workspace" % (self.path, name))
            if not os.path.isfile(p):
                raise WorkspaceError("%s: no prompt fragment %r" % (self.path, name))
            self._fragments[name] = _read_text(p)
        return self._fragments[name]

    def tool_description(self, name):
        if name not in self._descriptions:
            p = os.path.join(self._file("tool_descriptions"), "%s.yaml" % name)
            if not _inside(p, os.path.realpath(self.path)):
                raise WorkspaceError("%s: tool description %r resolves outside the workspace" % (self.path, name))
            if not os.path.isfile(p):
                raise WorkspaceError("%s: no tool description for %r" % (self.path, name))
            doc = yaml.safe_load(open(p, encoding="utf-8")) or {}
            if doc.get("name") != name:
                raise WorkspaceError("%s: %s describes %r" % (self.path, os.path.basename(p), doc.get("name")))
            self._descriptions[name] = doc
        return self._descriptions[name]

    def check(self, tool_names):
        """Every tool the environment exposes must be described, every Python file must keep to the isolation rules
        (isolation_problems), and the semantic_budget allocation must fit the shell's total with known consumers
        (harness/semantic_budget.allocation_problems). -> list of problems (empty = loadable)."""
        problems = []
        try:
            problems += isolation_problems(self.path)
        except WorkspaceError as e:
            problems.append(str(e))
        for n in tool_names:
            try:
                self.tool_description(n)
            except WorkspaceError as e:
                problems.append(str(e))
        try:
            if not os.path.isfile(self._file(self.key("agent_program"))):
                problems.append("%s: agent program %s missing" % (self.path, self.manifest[self.key("agent_program")]))
        except WorkspaceError as e:
            problems.append(str(e))
        for rel in (self.manifest.get("policies") or []):
            try:
                if not os.path.isfile(self._path(rel)):
                    problems.append("%s: policy file %s missing" % (self.path, rel))
            except WorkspaceError as e:
                problems.append(str(e))
        try:
            has_obligations = bool(self.completion_obligations_file())
        except WorkspaceError as e:
            problems.append(str(e))
            has_obligations = False
        if has_obligations:
            from harness.obligation_gate import CompletionObligationsError, load_completion_obligations
            try:
                load_completion_obligations(self.path)
            except CompletionObligationsError as e:
                problems.append(str(e))
        for key in ("middleware", "kernel_middleware"):
            try:
                for rel, _cls in self._entries(key):
                    if not os.path.isfile(self._path(rel)):
                        problems.append("%s: %s file %s missing" % (self.path, key, rel))
            except WorkspaceError as e:
                problems.append(str(e))
        try:
            from harness.semantic_budget import COMPONENT_CONSUMERS, allocation_problems, shell_total
            classes = {key: [cls for _rel, cls in self._entries(key)] for key in COMPONENT_CONSUMERS}
            problems += ["%s: semantic_budget: %s" % (self.path, p) for p in
                         allocation_problems(self.manifest.get("semantic_budget"), shell_total(), classes)]
        except WorkspaceError as e:
            problems.append(str(e))
        return problems

    def _entries(self, key):
        """(file, Class) per `<file>:<Class>` entry of a manifest list, in order."""
        out = []
        for entry in (self.manifest.get(key) or []):
            rel, _, cls = str(entry).partition(":")
            if not rel or not cls:
                raise WorkspaceError("%s: %s entry %r is not <file>:<Class>" % (self.path, key, entry))
            out.append((rel, cls))
        return out

    def module(self, rel):
        """This workspace's file `rel` as the module the shell and workspace code run (the loader's one copy)."""
        return _module(self._path(rel))

    def middleware_classes(self):
        """The middleware classes the shell's middleware host calls, in manifest order."""
        return [self.load(rel, cls) for rel, cls in self._entries("middleware")]

    def policy_files(self):
        """The clinical policy modules the shell composes after its own (runner/harness/engines/policy.py)."""
        return [self._path(rel) for rel in (self.manifest.get("policies") or [])]

    def completion_obligations_file(self):
        """The completion-obligation content the shell's obligation engine composes for every reader of a task's
        completion conjunction (runner/harness/obligation_gate.py), or None: then no task type has a conjunction."""
        rel = self.manifest.get("completion_obligations")
        return self._path(rel) if rel else None

    def kernel_middleware(self):
        """The capability classes the harness kernel runs after the fixed shell's own, in manifest order."""
        return [self.load(rel, cls) for rel, cls in self._entries("kernel_middleware")]

    def program_class(self):
        key = self.key("agent_program")
        self._file(key)
        return self.load(self.manifest[key], "AgentProgram")


def _read_text(path):
    """A workspace text file without the one trailing newline a text file ends with."""
    text = open(path, encoding="utf-8").read()
    return text[:-1] if text.endswith("\n") else text


def agent_route():
    """(base, key) the agent model is called with. The agent may be served by its own endpoint (MH_AGENT_BASE +
    MH_AGENT_KEY_FILE) while the judges and the VLM stay on MH_OPENAI_BASE -- e.g. an agent model reached through a
    relay. Unset, it is MH_OPENAI_BASE and the usual key (transport.load_key). The key file may list several keys, one
    per line: each agent draws one at random, so the load and the quota are spread over them, and a task re-run after
    a dead key left it unanswered can draw another."""
    import transport
    base = agent_bases()[0]
    keys = agent_keys()
    if not keys:
        return base, transport.load_key()
    return base, random.choice(keys)


def agent_bases():
    """The agent's endpoints in order: MH_AGENT_BASE may name several, comma-separated, that serve the same keys
    (some relays offer several host names, any of which may drop connections for minutes at a time). Falls back to
    MH_OPENAI_BASE, then a built-in default; a trailing /v1 is stripped. The first is used; WorkspaceAgent._chat moves
    to the next when a turn gave up on the one it is on."""
    raw = os.environ.get("MH_AGENT_BASE") or os.environ.get("MH_OPENAI_BASE") or "https://api.openai.com"
    out = []
    for b in raw.split(","):
        b = b.strip().rstrip("/")
        if b.endswith("/v1"):
            b = b[:-3].rstrip("/")
        if b and b not in out:
            out.append(b)
    return out


def agent_keys():
    """Every key of MH_AGENT_KEY_FILE, one per line; [] when the agent has no key file of its own."""
    kf = os.environ.get("MH_AGENT_KEY_FILE")
    if not kf:
        return []
    with open(os.path.expanduser(kf)) as fh:
        return [k.strip() for k in fh.read().splitlines() if k.strip()]


class WorkspaceAgent(object):
    """The runner's agent object for a workspace: the model configuration and transport stay here (shell);
    everything that decides what is said to the model is the workspace's.

    The program is the workspace's one for the model's protocol (runner/models.yaml) and is handed the shell's chat
    for that protocol: function calling chat(messages, tools_schema) -> assistant message (transport.chat_fc), text
    chat(messages, max_new_tokens) -> reply text (transport.chat_text). A model the table does not list raises
    models.UnknownModelError before anything else is built."""
    name = "gpt5"

    def __init__(self, task, path):
        import models
        import transport
        self.model = models.agent_model()
        self.protocol = models.protocol(self.model)
        self.reasoning = os.environ.get("MH_OPENAI_REASONING", "high")
        self.base, self._key = agent_route()
        self._spare_keys = [k for k in agent_keys() if k != self._key]
        random.shuffle(self._spare_keys)
        self._spare_bases = [b for b in agent_bases() if b != self.base]
        self.workspace = Workspace(path, self.protocol)
        tools = [t.get("name") for t in (task.get("available_tools") or []) if t.get("name")]
        problems = self.workspace.check(tools)
        if problems:
            raise WorkspaceError("; ".join(problems))
        chat = self._chat if self.protocol == FUNCTION_CALLING else self._chat_text
        self.program = self.workspace.program_class()(public_task(task), self.workspace, chat)

    def _chat(self, messages, tools_schema):
        """One FC turn. A key the relay refuses for good (out of quota, expired, invalid) is dropped for the rest of the
        task and the same turn is sent again with another key of the key file, so a dead key does not void the task."""
        import transport
        while True:
            msg = transport.chat_fc(messages, tools_schema, self.model, self.base, self._key, self.reasoning)
            while msg.get("key_refused") and self._spare_keys:
                self._key = self._spare_keys.pop()
                sys.stderr.write("[FC-KEY-SWITCH %d key(s) left]\n" % len(self._spare_keys))
                sys.stderr.flush()
                msg = transport.chat_fc(messages, tools_schema, self.model, self.base, self._key, self.reasoning)
            # the turn gave up on this endpoint for a reason that is not the key (connection dropped, timed out,
            # overloaded, after transport's own retries): the same turn goes to the next endpoint serving these keys
            if (msg.get("content") == "API_BRAIN_ERROR" and not msg.get("tool_calls") and not msg.get("key_refused")
                    and self._spare_bases):
                self.base = self._spare_bases.pop(0)
                sys.stderr.write("[FC-BASE-SWITCH -> %s, %d left]\n" % (self.base, len(self._spare_bases)))
                sys.stderr.flush()
                continue
            break
        unreachable = msg.pop("unreachable", None)
        if msg.pop("key_refused", None):
            # every key of the key file is refused for good (quota, expiry): no later task can run either, so a line is
            # appended to MH_QUOTA_STOP_FILE, on which the job's watchdog cancels the job, rather than turning each remaining task
            # into an API_BRAIN_ERROR that is judged, voided and re-run
            stop = os.environ.get("MH_QUOTA_STOP_FILE")
            if stop:
                import time
                try:
                    with open(stop, "a") as fh:
                        fh.write("%s all agent keys refused (%s)\n" % (time.strftime("%Y-%m-%dT%H:%M:%S"), self.model))
                except OSError:
                    pass
        elif unreachable:
            # every endpoint of the agent failed to answer through transport's whole retry budget (e.g. a TLS host-name
            # mismatch): an outage, which would turn every remaining task into a voided API_BRAIN_ERROR. The job stops the same way
            # and can be resumed when the gateway is back.
            stop = os.environ.get("MH_QUOTA_STOP_FILE")
            if stop:
                import time
                try:
                    with open(stop, "a") as fh:
                        fh.write("%s agent gateway unreachable (%s)\n" % (time.strftime("%Y-%m-%dT%H:%M:%S"), self.base))
                except OSError:
                    pass
        return msg

    def _chat_text(self, messages, max_new_tokens=400):
        import transport
        return transport.chat_text(messages, self.model, self.reasoning, max_new_tokens)

    @property
    def _protocol_variant_last(self):
        """The near-miss markup a text program repaired its last tool call from (run.py records it on the tool_call
        event, then resets it); None for a program that does not report one."""
        return getattr(self.program, "protocol_variant_last", None)

    @_protocol_variant_last.setter
    def _protocol_variant_last(self, value):
        if hasattr(self.program, "protocol_variant_last"):
            self.program.protocol_variant_last = value

    def act(self, state):
        return self.program.act(state)
