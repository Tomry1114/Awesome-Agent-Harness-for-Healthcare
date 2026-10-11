# Harness Design Principles

Every harness component encodes an assumption about the model. **Classify each mechanism by the assumption
it makes, because that determines its lifespan.** A harness that does not know which of its parts are
fragile will silently rot as models improve.

## The three layers

1. **Infrastructure layer — assumes "what capability INTERFACE the model needs."**
   Environment-facing, not model-facing. A failed portal submit, a patient-identity boundary, an
   irreversible write — these are facts about the WORLD; a smarter model does not make them disappear.
   STABLE. This is the harness's permanent value.

2. **Compensation layer — assumes "what the model CAN'T do."**
   Patches a model weakness. FRAGILE: as the model improves past the weakness the component fires less and
   becomes vestigial. Compensating for a weakness the model no longer has yields ZERO effect by
   construction (not an implementation failure — a category error of placement). MUST be removable.

3. **Capability-amplification layer — assumes "the better WAY to use the model's existing capability now."**
   The capability is present; this is a strategy choice about how to deploy it. SUBTLE & PROVISIONAL: the
   optimal implementation changes with each model generation (a stronger single-pass model self-corrects,
   shrinking an external loop's marginal value). Re-measure and re-tune it for each model; never hard-wire
   it into the core.

## Classification of this harness

| Mechanism | Layer | Rationale | Where |
|---|---|---|---|
| Subject binding + evidence ledger (`ScopeEvidenceBinding`) | **Infra** | environment-facing identity + audit interface | `runner/harness/capabilities/scope_evidence.py` |
| Commit-point / obligation PREREQUISITE gating | **Infra** | irreversible-write safety | `runner/harness/capabilities/obligation_lifecycle.py` |
| Process-output / commit integrity (`unresolved_operational_commit`; an unknown/failed write is not finalized over) | **Infra** | a failed env write is a world-fact | `runner/harness/capabilities/verify_commit.py` |
| Pre-commit redundant-write BLOCK; unknown-commit-state ESCALATE / RECONCILE | **Infra** | env write integrity | `runner/harness/capabilities/verify_commit.py`, `runner/harness/reconcile.py` |
| Reference integrity (a commit may only reference resources the agent retrieved or the public task names) | **Infra** | an ungrounded reference makes the substrate reject the commit | `runner/harness/capabilities/reference_integrity.py` |
| Evidence-contradiction veto (the final-answer adequacy audit's localized hard defect) | **Compensation** | the agent asserting something its own evidence contradicts | `runner/harness/capabilities/verify_commit.py` |
| `no_new_progress` loop-detection | **Compensation** | the agent repeating the same semantic action without progress | `runner/harness/capabilities/obligation_lifecycle.py` |
| Selective Epistemic Repair (adequacy audit + candidate B + conservative A/B select) | **Amplification** | PROVISIONAL — its value shrinks as single-pass self-correction improves | audit in `verify_commit.py`; repair and selection in `workspaces/medcta/h0/middleware/answer_repair.py`, `answer_candidate.py` |

## Design rules

1. **Tag every mechanism by layer** in its docstring (`# LAYER: infra|compensation|amplification`).
2. **Keep compensation removable** — it should come out cleanly when models outgrow it.
3. **Invest durably only in infrastructure** — that is the harness's permanent contribution.
4. **Treat amplification as provisional** — measure it for each model generation; re-tune (tau, critique
   form) or remove; never assume it persists.
5. **Let the harness self-report fragility** — log whether each mechanism FIRES and whether it CHANGES an
   outcome. A compensation mechanism whose fire-rate has gone to ~0 is announcing that it is now vestigial.

## What the harness provides

The durable contribution is not "the harness raises outcome" — that is a compensation property that fades
as models improve. It is:

> **Environment-facing safety + commit integrity (infrastructure)** without sacrificing outcome, plus a
> **pluggable, per-model-reassessed amplification layer**.

A near-zero outcome effect on a strong model is the expected signature of compensation against a model that
has already passed the weakness, not a sign that the infrastructure failed.

---

# Scoped Repair: one unified repair backbone across all three substrates

The harness does not only constrain safety. Its **outcome-facing** mechanism is a single backbone —
**RepairFinding → Scoped Repair lifecycle → Delta Validation** — shared by every substrate. A new dataset is
added by writing a *repair surface adapter* (`runner/harness/repair_surface.py`), not by adding kernel branches.

## Why a backbone and not per-dataset verifiers

An agent acts on harness findings, but a finding that names only a *vague, goal-level obligation* ("write a
triage note") can lead it to **overwrite already-substantive content** to look compliant. Once a finding is
delivered where the agent acts on it, the binding constraint is the finding's *content quality*. So every
finding must be **localized, minimal, and non-degrading** — identically for a form field, a FHIR path, or an
answer claim.

## The unit: RepairFinding

`{target_type, target_path, defect_type, operation, required_change, protected_paths,
preserve_requirements, allowed_capabilities}` (`runner/harness/repair.py`) — names exactly ONE concrete defect,
the smallest fix, and what must be preserved. A finding with no concrete target+change is dropped (no vague
REVISE). Stable `finding_id` = hash(task, rule, target, defect) → the dedup key that stops re-nagging.

## The lifecycle (ledger.repair_findings)

`OPEN → DELIVERED → ATTEMPTED → {RESOLVED | REGRESSED | EXHAUSTED}`. A delivered finding is NOT re-sent until
the agent changes the target (anti-churn; at most 2 repair attempts, then EXHAUSTED). Every attempt is
**delta-validated**.

## Delta validation (what makes it non-degrading)

`accept ⇔ target_resolved ∧ protected_content_preserved ∧ ¬new_conflict` (`runner/harness/repair_delta.py`).
Structured/numeric protected content must be EQUAL (a PB dose/route cannot drift); text must be retained
(APPEND keeps it, OVERWRITE fails → `repair_regression`). Reindex-tolerant for list targets: after a REMOVE,
protected items are matched by content membership, not position.

## Mapping the three repair surfaces

| Proposal surface | completeness criterion | repair operations |
|---|---|---|
| FORM (GUI portal) | required field present & sufficient | ADD / EDIT |
| FHIR (clinical API) | required path present & consistent | ADD / EDIT / REPLACE |
| ANSWER (perception) | every perceptual claim has an observation trace | REACQUIRE / VERIFY_OR_REMOVE / EDIT_OR_REMOVE |

The FORM/FHIR finding source is `goal_alignment` (Scoped Repair); the ANSWER finding source is
`evidence_coverage`. Both live in each benchmark's workspace (`workspaces/<bench>/h0/middleware/`).

---

# evidence_coverage: claim-conditioned observational coverage (the ANSWER gate)

On the perceptual substrate the characteristic failure is **not** an incomplete write — it is the agent
stating findings it never actually observed (wrong tool, wrong path, or fabrication). `verify_commit` is inert
without a commit; `evidence_coverage` covers this case through the SAME backbone.

## Naming discipline (avoid concept-smuggling)

It is **claim-conditioned observational coverage / perceptual traceability** — NOT "gold tool-path
completeness". There is no gold path, so it can establish only:

> every *perceptual* claim in the final answer traces to an actually-executed observation of its target

It cannot (and does not) claim the agent called every tool a correct decision needs. Use `claim-observation
coverage` in code and documentation; never `tool-path completeness`.

## Claims must be typed first

| claim_type | gated by evidence_coverage? |
|---|---|
| perceptual ("nodule in RLL") | YES — must have an observation trace |
| interpretive ("favors malignancy") | only requires ≥1 covered perceptual premise |
| background ("malignancy usually …") | NO |
| recommendation ("recommend biopsy") | NO |

This prevents flagging the diagnosis label itself as "no tool observed the diagnosis".

## Deterministic-first, judge at the margin

1. The ledger records a normalized observation per perception/read tool call
   `{subject, region, modality, attributes_observed, result_status}`.
2. Deterministic match: region never observed → finding; region+attribute observed → covered (no judge).
3. The judge is consulted ONLY when the region was looked at but the attribute is unclear (whole-image look) —
   "do the agent's own observations support this claim?"

Only deterministic defects (`unobserved_target`) are enforced. Semantic findings are advisory: they go to
candidate mode (keep answer A, ask for B, adopt B only if clearly better) and never force a claim deletion.

## Three defect cases (not always REACQUIRE)

| situation | defect_type | operation |
|---|---|---|
| target never observed | `unobserved_target` | REACQUIRE_EVIDENCE |
| observed, but observation doesn't support the claim | `unsupported_by_observation` | VERIFY_OR_REMOVE |
| claim names no concrete target | `untraceable_claim` | EDIT_OR_REMOVE |

A REACQUIRE finding with a routable affordance is executed as a read-only ACQUIRE (the runner runs the
perception tool), not returned as a REVISE that merely asks the agent to look again.

## The judge never names a tool — the affordance registry does

The judge says only *what observation is missing*. The kernel's **affordance registry**
(`runner/harness/affordance.py`) maps that need to *executable* tool names drawn from the task manifest
(`select_tools(available_tools, region, modality)`; `select_diversified_tool` avoids repeating a tool already
tried), so feedback can never suggest a non-existent / wrong-affordance tool. No match → no tool suggested
(silent beats hallucinated).

---

# Layer by MECHANISM, not by file

A single capability can contain mechanisms of different layers. Classify each mechanism, not the file.

| Capability | Infrastructure part | Amplification / Compensation part |
|---|---|---|
| `repair_delta` | non-degrading delta validation (deterministic) | — |
| `evidence_coverage` | deterministic observation↔claim coverage; affordance registry | claim decomposition/classification; margin semantic support (advisory, candidate mode) |
| `goal_alignment` (Scoped Repair) | the lifecycle + dedup + delta | L2 semantic sufficiency (judge locates the defect) |
| `obligation_lifecycle` | obligation state machine; prerequisite gating; dedup; retry budget | `no_new_progress` loop-detection (Compensation) |
| `scope_evidence` | subject binding; wrong-scope BLOCK; evidence ledger | — |
| `verify_commit` | commit verification; redundant-commit BLOCK; unknown state → RECONCILE/ESCALATE | evidence-contradiction veto (Compensation); the adequacy audit it records feeds the Amplification repair in the workspace |

Only a *strong* observational gap (a perceptual claim with zero observation) is allowed to force a REVISE.
Blocking a final on "a possibly-relevant tool was unused" pushes an agent into over-checking loops, so such
signals are recorded, not enforced.

---

# Framework

```
                       ORACLE-BLIND boundary  (never reads gold / reference / checkpoints)
 ----------------------------------------------------------------------------------------
                                  AGENT  (brain: weak..strong)
       propose action -> execute -> observe -> ... -> final answer
             |                          ^                  |
   KERNEL    v before_action           | after_action     v before_final     (3 commit points)
   capabilities, layered by MECHANISM:
     INFRASTRUCTURE   subject_binding | verify_commit (+RECONCILE) | reference_integrity | repair_delta |
                      obligation state-machine/dedup/retry |
                      evidence_coverage: deterministic observation coverage | affordance_registry
     AMPLIFICATION    scoped_repair L2 (judge locates defect) |
                      evidence_coverage: claim decomposition / semantic support |
                      required_context / active_evidence (read-only acquisition) |
                      answer repair + conservative A/B selection (MedCTA)
     COMPENSATION     evidence-contradiction veto | no_new_progress loop-detection
             |
             v  combine(ESCALATE > BLOCK > REVISE > RECONCILE > ALLOW)
     SCOPED REPAIR core (shared by all substrates):
        finding sources -> RepairFinding{target, defect, op, protected, allowed_caps}
           scoped_repair (FORM / FHIR)    evidence_coverage (ANSWER claim)
           -> ledger lifecycle: dedup -> delivered -> attempted -> DELTA-validate ->
              RESOLVED / REGRESSED(veto) / EXHAUSTED
           -> repair surface adapter (the ONLY substrate-specific code): FORM | FHIR | ANSWER
     feedback render:  localized patch protocol; tool names from the affordance registry, NOT the judge
             |
             v  back to AGENT  (ADD / EDIT / REMOVE / REACQUIRE …, preserve substantive content)

 LEDGER (harness-external state, agent cannot mutate):
   active_subject | observations{subject,region,modality,attrs,status} | evidence | obligations |
   commit_history | completed_commits | repair_findings(lifecycle) | opportunities(denominators)
```

The kernel and its capabilities are the fixed shell (`runner/harness/`). Each benchmark's default workspace
(`workspaces/<bench>/h0/manifest.yaml`) adds `kernel_middleware` (run by the kernel after the shell's own
capabilities), host `middleware` (called by `runner/middleware_host.py`), and policy modules.

---

# Per-dataset mechanism roster

The sections above are the *principles*; this is the concrete, per-substrate roster of what is wired in each
default workspace. Type = **NUDGE** (hand the gap back, the agent does the work) / **DEFENSE** (block an
unsafe or degrading action) / **EXECUTE** (the harness itself performs a read-only step, or a bounded
completion, and verifies it).

## PhysicianBench (FHIR clinical — text note + orders)

Workspace: `workspaces/pb/h0/`.

| Mechanism | Type | What it does | Where |
|---|---|---|---|
| low-budget warning | NUDGE | with ≤8 executed actions left and the deliverable still missing, one warning to write it now | `middleware/deliverable.py` (`before_model`) |
| pre-final deliverable nudge | NUDGE | a final with the deliverable not yet written is turned back (up to 3×) with the exact required path | `middleware/deliverable.py` (`before_final`) |
| deliverable_freshness | NUDGE | a read-back-confirmed order, or newly retrieved evidence, after the last write → update the note (≤2) | `middleware/deliverable.py` (`before_final`) |
| specificity_completion | NUDGE | the note must CONCRETELY provide every slot the public task requires (value-free; the agent fills it from its own judgment and retrieval) (≤2) | `middleware/deliverable.py` (`before_final`) |
| commitment_ledger (append-only union) | NUDGE | parses the agent's OWN note; an order it presents as place-now / already-done with no read-back-confirmed order → re-enter (place it, or withdraw it citing a patient finding) (≤6). The union re-parses each round, so commitments added by a later rewrite are enforced too | `middleware/deliverable.py` (`before_final`) |
| retention | DEFENSE | snapshot version A when a correction nudge fires; a rewrite that DROPS grounded facts/topics A had is not executed and an additive patch is requested (≤2, fails open); emits `deliverable_retention_block` | `middleware/deliverable.py` (`before_tool`) |
| over-budget deliverable write | EXECUTE | the one exact deliverable write may still run after the tool budget is spent; at episode end one forced write turn is given if the file is still missing | `middleware/deliverable.py` (`exempt_from_budget`, `after_episode`) |
| chart review obligations | DEFENSE | goal-conditional evidence obligations (Patient / Condition / Observation) that must be CHECKED before the note is written | `policies/chart_review.yaml` |
| required_context | EXECUTE (read-only) | an unresolved evidence obligation of the matched commit point is acquired with the adapter's exact query before the commit; ESCALATE for an irreversible commit when it cannot be acquired | `middleware/required_context.py` |
| fhir_empty_targeted_search_broad_reacquire | EXECUTE (read-only) | a restrictive FHIR search that returned nothing → one broad patient-level read (≤2) | `middleware/fhir_empty_targeted_search_broad_reacquire.py` |
| goal_alignment (Scoped Repair) | NUDGE | localized, delta-validated repair findings on a create/update/submit | `middleware/goal_alignment.py` |
| commitment_execution | NUDGE | the deliverable states an action is being performed but no matching commit was executed | `middleware/commitment_execution.py` |

`active_evidence` and `evidence_coverage` are listed in the manifest but are inert on FHIR (perception tools
only / the final answer is not the graded commit). On PB the harness itself performs only read-only
acquisition; it never writes orders or note content for the agent.

## HealthAdminBench (GUI/form — Epic-like portal, irreversible submit)

Workspace: `workspaces/hab/h0/`.

| Mechanism | Type | What it does | Where |
|---|---|---|---|
| recovery (Bounded Clinical Recovery) | EXECUTE | at the agent's final (and once after a forced stop): navigate, fill what the agent's own values or persisted records bind, acquire skipped evidence reads, submit only where the agent began the submission, then verify against the task's completion conjunction (`verify_effect`). A Gap Router picks one path per episode (evidence / execution / verification / decision). A field that needs new content or a new decision → BLOCK and hand back to the agent (re-entry, ≤2 rounds); an empty form is never submitted by the harness (`empty_form_no_submit`); recovery stops re-engaging after 2 consecutive blocked invocations | `middleware/recovery.py`, `recovery/` |
| step-kind reader gate | DEFENSE | inside recovery, READ / NAVIGATE / ACQUIRE / VERIFY steps mint no mutation authorization; only STAGED_WRITE / IRREVERSIBLE_COMMIT are writes, and an irreversible commit is never repeated after an unknown outcome | `recovery/contracts.py`, `recovery/kernel.py` |
| obligation_nudge | NUDGE | do not accept a premature final while the task's completion obligations are unmet (≤3) | `middleware/obligation_nudge.py`, `policies/completion_obligations.yaml`, `runner/harness/obligation_gate.py` |
| grounding_audit | NUDGE | the authored note must reference salient facts the agent itself observed (codes, IDs, fax numbers) (≤2) | `middleware/grounding_audit.py` |
| read_acquire | EXECUTE (read-only) | when the current page shows an unmet public read-only evidence control, one read-only click; a flag counts as done only when the live state flips (≤2 attempts) | `middleware/read_acquire.py` |
| precommit_audit | DEFENSE | a terminal commit does not run while a required control on the page is empty; names the fields (≤3), then tells the agent not to submit an incomplete form | `middleware/precommit.py` |
| attachment check | DEFENSE | on the fax portal, a send with files downloaded but nothing attached is refused (≤2) | `middleware/attachment.py` |
| collection_nudge | NUDGE | the agent observed more distinct diagnosis/CPT codes than the form holds → reminder naming only the concept and counts (≤2) | `middleware/collection_nudge.py` |
| write_verify | NUDGE | after a fill/select, the read-back receipt shows the value did not land, or landed in a control of a different field concept (≤4) | `middleware/write_verify.py` |

The kernel middleware list (`goal_alignment`, `required_context`, `active_evidence`, `evidence_coverage`,
`commitment_execution`, `fhir_empty_targeted_search_broad_reacquire`) is shared with the other workspaces.

## MedCTA (perceptual clinical reasoning — tool sandbox, VLM tools, continuous GAcc)

Workspace: `workspaces/medcta/h0/`. Outcome is the upstream Goal-Accuracy judge
(`runner/adapters/medcta/gacc_judge.py`).

| Mechanism | Type | What it does | Where |
|---|---|---|---|
| final-answer adequacy audit | DEFENSE | one judge audit of the final answer for a commit point with a claim-support postcondition; a localized hard defect is vetoed (REVISE `evidence_contradiction`); the audit is recorded read-only as `ctx.final_audit` | `runner/harness/capabilities/verify_commit.py`, `runner/policies/clinical/evidence_grounding.yaml` |
| answer_repair | NUDGE | repair policy on that audit: a repairable gap engages repair only when the answer does not address the task or the auditor is confident (≥0.7) | `middleware/answer_repair.py` |
| active_evidence (ACQUIRE) | EXECUTE (read-only) | elicit the agent's two most plausible interpretations + the single discriminating observable (region+attribute) from the answer and the PUBLIC question (`elicit_discriminator`), and acquire that observation read-only if it was never focus-observed | `middleware/active_evidence.py`, `runner/harness/engines/semantic.py` |
| evidence_coverage (ANSWER gate) | infra+ampl | every perceptual claim must trace to an actually-executed observation of its target; deterministic-first, judge only at the margin; the affordance registry names the tool | `middleware/evidence_coverage.py` |
| answer_candidate (conservative A/B selector) | DEFENSE | after an ACQUIRE or candidate REVISE, the earliest answer is held as A and the next final is candidate B; B replaces A only if the judge prefers it by a margin ≥ tau with the critique resolved and no new hard violation. A flip that only re-interprets the SAME finding without independent resolving evidence is ambiguous → A is kept. Bounded rounds (manifest `semantic_budget.rounds`) | `middleware/answer_candidate.py` |

## Cross-substrate invariants (all three)

| Dimension | Shared design |
|---|---|
| oracle-blind | only the public instruction + the agent's own trajectory/note/retrieval/answer; NEVER a hidden checkpoint value / test name / dose / diagnosis / reference trace |
| content vs process | the harness fixes PROCESS (loops / premature give-up / mechanical closure / read-only acquire); it NEVER authors CONTENT (field values / clinical decisions) |
| three-gate lever test | ① is the value/action already in the trajectory? ② is the realization read-only / reversible / non-authoring? ③ does it need judgment? Only ① present + ② safe is a legitimate harness lever |
| two hard boundaries | the SUBMIT boundary (irreversible; the harness never submits from scratch) + the AUTHORING boundary (the harness never writes committed free text) |
| nudge vs execute | a NUDGE helps only if the agent acts on it; where a gap can be closed read-only, the harness performs the read itself (ACQUIRE) and the agent re-reasons with more evidence |
