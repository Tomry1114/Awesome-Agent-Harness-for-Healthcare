"""Harness-facing dataset-adapter contract (the harness side of the seam).

The runner talks to ONE object per dataset: its identity (env_type, benchmark name, capability manifest) and the
named runtime gates its substrate has (applies()). A dataset's helpful interventions are not adapter hooks: they
are workspace middleware (workspaces/<bench>/h0/middleware) called through the middleware host
(runner/middleware_host.py), or kernel middleware the harness kernel runs. Concrete per-dataset adapters live in
runner/adapters/ and subclass HarnessAdapter.

DESIGN INVARIANT (do not regrow a gate catalogue in the Runner or here): a new dataset gate is workspace middleware
on the host hook for its stage, never a new `if new_gate: adapter.new_gate()` Runner branch or adapter method.
"""


class HarnessAdapter(object):
    """Base harness-adapter contract. Subclass ONE per dataset (in runner/adapters/)."""

    env_type = None        # "fhir" | "gui" | "tool_sandbox"
    name = None            # canonical benchmark name
    manifest_yaml = None   # path (repo-relative) to the substrate capability manifest
    gates = frozenset()    # named runtime gates this dataset's substrate opts into (see applies())

    def applies(self, gate):
        return gate in self.gates

    def __repr__(self):
        return "<%s env=%s name=%s>" % (type(self).__name__, self.env_type, self.name)
