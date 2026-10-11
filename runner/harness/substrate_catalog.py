# -*- coding: utf-8 -*-
"""What a substrate IS, read from the adapter that dispatches it: its tool vocabulary, the writes that are its
deliverable, and the parameters its tools accept.

WHY THE ADAPTER IS THE TRUTH, not the static task file. The adapter is the thing that dispatches. A name
it does not expose cannot run however many task files mention it, and a name it exposes is runnable
whether or not a task file happens to list it. For PhysicianBench the task.json lists five GENERIC
names (fhir_create, fhir_read, fhir_search, get_lab_reference_range, write_file), while
PbNativeEnv._setup_native_registry OVERWRITES task["available_tools"] with the upstream PB ToolRegistry's
schemas the moment it starts.

UNAVAILABLE IS NOT EMPTY. If the adapter's vocabulary cannot be read (the upstream package is not
importable under this interpreter), these functions raise rather than fall back to the task file.
"""
import os
import sys


class SubstrateUnavailable(RuntimeError):
    """The adapter that defines this substrate's tools could not be consulted."""


def _adapters():
    """{env_type: adapter class}. The adapter is the thing that DISPATCHES, so it defines the substrate."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    # runner/
    if here not in sys.path:
        sys.path.insert(0, here)
    import environments as _E
    out = {}
    for name in dir(_E):
        cls = getattr(_E, name)
        et = getattr(cls, "type", None)
        if isinstance(cls, type) and isinstance(et, str) and et not in (None, "base"):
            out.setdefault(et, cls)
    return out


def _pb_native_names(cls):
    """PbNativeEnv rebuilds available_tools from the UPSTREAM registry at start-up, so its class-level
    IMPLEMENTED_TOOLS is a superset that includes names the registry does not publish. Read the registry
    when it can be reached; that is the list env.available_tools() will return."""
    return sorted({s["name"] for s in _pb_schemas() if s.get("name")})


def _pb_schemas():
    """The upstream registry's tool schemas -- what PbNativeEnv publishes and dispatches."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if here not in sys.path:
        sys.path.insert(0, here)
    import environments as _E
    up = getattr(_E, "_PB_UPSTREAM", None)
    if not up or not os.path.isdir(up):
        raise SubstrateUnavailable("upstream PhysicianBench package not found at %r" % (up,))
    if up not in sys.path:
        sys.path.insert(0, up)
    try:
        from agent.tool_registry import FHIR_TOOL_SCHEMAS, FILE_TOOL_SCHEMAS
    except Exception as exc:
        raise SubstrateUnavailable("upstream tool registry not importable: %r" % (exc,))
    return list(FHIR_TOOL_SCHEMAS) + list(FILE_TOOL_SCHEMAS)


def vocabulary(env_type="fhir"):
    """The tool names the ADAPTER exposes for this substrate. -> [str], sorted.

    Raises SubstrateUnavailable rather than falling back: a vocabulary that quietly used a different
    source would name tools the adapter cannot run.
    """
    adapters = _adapters()
    cls = adapters.get(env_type)
    if cls is None:
        raise SubstrateUnavailable("no adapter for env_type=%r (have %s)"
                                   % (env_type, sorted(adapters)))
    if env_type == "fhir":
        return _pb_native_names(cls)
    names = sorted(getattr(cls, "IMPLEMENTED_TOOLS", ()) or ())
    if not names:
        raise SubstrateUnavailable("adapter %s declares no tools" % (cls.__name__,))
    return names


def deliverable_writes(env_type):
    """{tool: {"path": arg, "content": arg}} for the tools that write this substrate's deliverable, or None
    when its adapter declares none. The runtime records the artifact from this and carries no list of its own.

    Raises SubstrateUnavailable for an env_type with no adapter, rather than answering None: "this substrate
    has no written deliverable" and "there is no such substrate" are different facts."""
    cls = _adapters().get(env_type)
    if cls is None:
        raise SubstrateUnavailable("no adapter for env_type=%r (have %s)" % (env_type, sorted(_adapters())))
    decl = getattr(cls, "DELIVERABLE_WRITES", None)
    return {t: dict(m) for t, m in decl.items()} if decl else None


def tool_parameters(env_type):
    """{tool: [parameter names]} as the substrate's published schemas declare them, or None where the adapter
    publishes names only. Used to pick the argument a harness-built action names its subject through: one the
    tool actually accepts."""
    cls = _adapters().get(env_type)
    if cls is None:
        raise SubstrateUnavailable("no adapter for env_type=%r (have %s)" % (env_type, sorted(_adapters())))
    if env_type != "fhir":
        return None
    out = {}
    for s in _pb_schemas():
        f = s.get("function", s)
        if f.get("name"):
            out[f["name"]] = sorted(((f.get("parameters") or {}).get("properties") or {}).keys())
    return out
