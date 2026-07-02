"""Bounded Clinical Recovery v3 - Benchmark Adapters (Layer 4).

Task/field/lifecycle/state-path NORMALIZATION only. A benchmark adapter maps one dataset's task shape into the
kernel/workflow vocabulary (schema, authoritative_state, system_metadata, committed goals, lifecycle triggers,
concrete state-paths). It does NOT change any kernel or workflow rule.

Benchmark adapter files are the ONLY layer permitted to carry dataset-specific task knowledge. They remain
oracle-blind at runtime (never read gold/checkpoint/reference-trace).
"""
