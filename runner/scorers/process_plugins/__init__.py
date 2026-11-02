"""Offline process extensions, separate from runtime benchmark semantics.

Each non-private module exports PLUGIN. Discovery happens only when an offline
report is built; these hooks are never registered in the agent/harness runtime.
"""
