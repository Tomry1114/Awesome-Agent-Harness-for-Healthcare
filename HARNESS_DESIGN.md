# Harness Design

The harness checks agent actions, tracks evidence and pending obligations, and supports bounded repairs. Shared execution controls live in `runner/harness/`; benchmark-specific components are configured in `workspaces/<benchmark>/h0/manifest.yaml`.

## Execution

The kernel runs checks before an action, after its result, and before the final answer. Capabilities check patient scope, prerequisites, resource references, and commit state. Decisions can allow execution, request a revision or read-only acquisition, block an action, or reconcile an uncertain result.

The ledger records evidence, obligations, commits, and repair attempts. Host middleware handles benchmark-specific interactions around model calls, tools, and final answers. Kernel middleware adds checks through the workspace manifest.

## Repair

A `RepairFinding` identifies a concrete target, defect, requested change, and content to preserve. The ledger tracks delivery and attempts, deduplicates findings, and bounds retries. Delta validation checks whether the defect was resolved without damaging protected content or introducing conflicts.

Repair surfaces map this process to portal fields, FHIR resource paths, and answer claims. For perceptual answers, evidence coverage checks whether claims can be traced to executed observations; it does not establish diagnostic correctness.

## Benchmark components

| Benchmark | Main behavior |
|---|---|
| PhysicianBench | Checks chart-review obligations, requests missing or updated deliverables, tracks order commitments, and acquires missing context through read-only queries. |
| HealthAdminBench | Checks required fields and attachments, verifies form writes, and attempts bounded recovery using the agent's values or persisted records. |
| MedCTA | Checks answer support, acquires additional observations, and compares a revised answer with the original before replacing it. |

## Boundaries

Runtime checks use the public task and observed execution, rather than hidden answers or grading checkpoints. Components can request content changes from the agent; they do not supply new clinical decisions or author committed free text. Recovery must respect the agent's existing intent, write authorization, and uncertain commit state.

Model-dependent repair and prompting strategies remain workspace components so they can be changed or removed independently of execution controls.

See the [extension guide](runner/README.md#extension-guide) for component registration and hook signatures.
