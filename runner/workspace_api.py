# -*- coding: utf-8 -*-
"""The one shell module workspace code may import.

A workspace module (agent program, host or kernel middleware, a module loaded through load_relative) imports from the
runner only the names listed in __all__ below, and the standard library only as runner/workspace.py STDLIB_IMPORTS
allows. runner/workspace.py enforces both twice: at load time (every workspace module runs with builtins whose
__import__ admits only these, and an import of this module yields a read-only facade over __all__), and statically
(Workspace.check, isolation_problems).

What is here is what the default (H0) workspaces need, and nothing reaches the shell's state through it:
    results the middleware host applies       Observation, Revision, Nudge, PostRevision, Read, Repair, Accept
    workspace-relative loading                load_relative, workspace_of
    kernel middleware base and verdict words  Capability, decision (ALLOW ... ESCALATE)
    pure shell functions over values          the rest: they read their arguments and the shell's fixed content
                                              (completion obligations of a workspace, the published tool schemas),
                                              never the environment, a ledger, a trajectory or a judge of their own
Functions that take a judge take the metered judge a context hands out (ctx.semantic_check(), ctx.budgeted_judge(),
ctx.judge). Container constants are copies; a set is a tuple in the set's iteration order.

A name is exported here only when workspace code needs it; a module object is never exported (its
attributes would reach every module it imports).
"""
import copy as _copy
import types as _types

import middleware_host as _mh
import workspace as _ws
from harness import actuation as _actuation
from harness import adapter_compiler as _adapter_compiler
from harness import affordance as _affordance
from harness import bindings as _bindings
from harness import capability as _capability
from harness import decision as _decision
from harness import evidence_state as _evidence_state
from harness import gap as _gap
from harness import obligation_gate as _obligation_gate
from harness import observation as _observation
from harness import repair as _repair
from harness import repair_delta as _repair_delta
from harness import repair_surface as _repair_surface
from harness import result_facts as _result_facts
from harness import risk as _risk
from harness import semantics as _semantics
from harness import substrate_catalog as _substrate_catalog
from harness.engines import semantic as _semantic
from adapters.hab import collection as _collection
from adapters.hab import common as _common
from adapters.hab import gui_control_model as _gui

_EXPORTS = {}


def _export(name, value):
    if isinstance(value, (dict, list)):
        value = _copy.deepcopy(value)
    elif isinstance(value, (set, frozenset)):
        value = tuple(value)
    _EXPORTS[name] = value


def _late(module, src, fn):
    """A shell function as workspace code calls it: resolved in its module at call time, so the workspace calls what the
    shell module holds then (the function itself, or a test's stand-in for it), the same late binding a lazy import
    inside the calling function would give."""
    def call(*args, **kwargs):
        return getattr(module, src)(*args, **kwargs)
    call.__name__, call.__qualname__, call.__doc__ = fn.__name__, fn.__qualname__, fn.__doc__
    return call


def _names(module, names):
    """Export each name of `module`; a (shell name, exported name) pair exports it under a public name. A plain function
    is exported late-bound (_late); a class or a constant as it is (a container constant as a copy)."""
    for name in names:
        src, dst = name if isinstance(name, tuple) else (name, name)
        value = getattr(module, src)
        _export(dst, _late(module, src, value) if isinstance(value, _types.FunctionType) else value)


# ---- results the middleware host applies (runner/middleware_host.py)
_names(_mh, ("Observation", "Revision", "Nudge", "PostRevision", "Read", "Repair", "Accept"))
# ---- workspace-relative loading (runner/workspace.py)
_names(_ws, ("load_relative", "workspace_of"))
# ---- kernel middleware (runner/harness/capability.py, decision.py, affordance.py)
_names(_capability, ("Capability",))
_export("decision", _ws.namespace("decision", {n: getattr(_decision, n) for n in
                                               ("ALLOW", "RECONCILE", "ACQUIRE", "REVISE", "DEFER_FINAL", "BLOCK",
                                                "ESCALATE")}))
_export("affordance", _ws.namespace("affordance", {n: _late(_affordance, n, getattr(_affordance, n)) for n in
                                                   ("select_tools", "select_diversified_tool")}))
# ---- pure shell functions and constants
_names(_actuation, ("next_action_of",))
_names(_semantic, ("classify_evidence_need", "detect_unexploited_entity_label", "elicit_discriminator",
                   "grounded_verification_query", "synthesis_grounding_check", "decompose_claims",
                   "claim_semantic_support", "scoped_goal_findings", "select_relevant_obligations",
                   "verify_claim_support", ("_as_bool", "as_bool"), ("_format_evidence", "format_evidence"),
                   ("_json_obj", "json_obj")))
_names(_observation, (("Observation", "EvidenceObservation"), ("_n", "normalized_text"), "observation_reliability",
                      "RELIABILITY_CONFIDENT", "region_observed", "attribute_observed", "coverage_findings"))
_names(_repair, ("enforceable", "RepairFinding"))
_names(_result_facts, ("result_is_empty",))
_names(_repair_delta, ("validate_repair",))
_names(_repair_surface, ("surface_for", "is_present", "target_sig", "effect_paths", "effect_fingerprint"))
_names(_risk, ("at_least", "R2"))
_names(_adapter_compiler, ("compile_evidence_request",))
_names(_evidence_state, ("is_resolved",))
_names(_gap, ("MISSING_CONTEXT",))
_names(_semantics, ("same_subject", "canonicalize"))
_names(_substrate_catalog, ("tool_parameters",))
_names(_bindings, ("classify_field", "SEMANTIC", "SEMANTIC_SOURCE_PRIORITY", "OPERATIONAL_SOURCE_PRIORITY",
                   "AGENT_COMMITMENT", "AUTHORITATIVE_STATE", "BOUND_EVIDENCE", "SYSTEM_METADATA"))
_names(_obligation_gate, ("action_control", "completion_conjunction", "conjunct_met", "decision_of", "evaluate",
                          "obligation_view", "readback_portals"))
_names(_gui, ("is_control_complete", "is_terminal_action", "norm_value", "FIELD_ALIASES",
              ("_field_concept", "field_concept"), ("_is_lookup_section", "is_lookup_section")))
_names(_common, ("CPT_HCPCS_RE", "ICD10_RE", ("agent_actions", "portal_agent_actions")))
_names(_collection, ("next_collection_action", "present_values", "committed_collection", "COLLECTION_CONCEPTS"))

globals().update(_EXPORTS)
__all__ = tuple(sorted(_EXPORTS))
