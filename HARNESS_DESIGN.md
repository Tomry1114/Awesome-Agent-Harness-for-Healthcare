# Harness Design Principles

Every harness component encodes an assumption about the model. **Classify each mechanism by the assumption
it makes, because that determines its lifespan.** A harness that does not know which of its parts are
fragile will silently rot as models improve.

## The three layers

1. **Infrastructure layer — assumes "what capability INTERFACE the model needs."**
   Environment-facing, not model-facing. A failed portal submit, a patient-identity boundary, an
   irreversible write — these are facts about the WORLD; a smarter model does not make them disappear.
   STABLE. Invest durably here. This is the harness's permanent value.

2. **Compensation layer — assumes "what the model CAN'T do."**
   Patches a model weakness. FRAGILE: as the model improves past the weakness the component fires less and
   becomes vestigial. Compensating for a weakness the model no longer has yields ZERO effect by
   construction (not an implementation failure — a category error of placement). MUST be ablatable.

3. **Capability-amplification layer — assumes "the better WAY to use the model's existing capability now."**
   The capability is present; this is a strategy choice about how to deploy it. SUBTLE & PROVISIONAL: the
   optimal implementation keeps evolving with each model generation (a stronger single-pass model
   self-corrects, shrinking an external loop's marginal value). Treat as temporary; re-measure and re-tune
   each generation; never hard-wire into the core.

## Classification of THIS harness (measured on gpt-5.5, not asserted)

| Mechanism | Layer | Evidence / fragility | Ablation control |
|---|---|---|---|
| Subject binding + evidence ledger (ScopeEvidenceBinding) | **Infra** | environment-facing identity + audit interface | always on |
| Commit-point / obligation PREREQUISITE gating | **Infra** | irreversible-write safety | always on |
| Process-output / commit integrity (unresolved_operational_commit; unknown/failed write not finalized over) | **Infra** | a failed env write is a world-fact | `MH_REPAIR != none` |
| Pre-commit redundant-write BLOCK; unknown-commit-state ESCALATE | **Infra** | env write integrity | always on |
| Grounding / claim-support contradiction veto | **Compensation** | fired **2/43** tasks on gpt-5.5 — near-vestigial | `MH_REPAIR`: `hard`+ enforces it, `none` reverts to advisory |
| `no_new_progress` loop-detection | **Compensation** | **never fired** on gpt-5.5 (it does not doom-loop) | (default on; keep flaggable) |
| Selective Epistemic Repair Loop (adequacy audit + candidate B + conservative A/B select) | **Amplification** | PROVISIONAL — value shrinks as single-pass self-correction improves | `MH_REPAIR = soft|select|full` |

## Design rules

1. **Tag every mechanism by layer** in its docstring (`# LAYER: infra|compensation|amplification`).
2. **Make compensation ablatable** — removable in one flag when models outgrow it.
3. **Invest durably only in infrastructure** — that is the harness's permanent contribution.
4. **Treat amplification as provisional** — measure it each model generation; re-tune (tau, critique form)
   or remove; never assume it persists.
5. **Let the harness self-report fragility** — log whether each mechanism FIRES and whether it CHANGES an
   outcome. A compensation mechanism whose fire-rate has gone to ~0 is announcing it is now vestigial and
   should be retired. (We already saw this: grounding-veto 2/43, loop-detection 0/43.)

## The honest claim (what to put in the paper / hand to the next model generation)

The durable contribution is NOT "the harness raises outcome" — that is a fragile compensation narrative that
evaporates with the next model. It is:

> The harness provides **environment-facing safety + commit integrity (infrastructure)** without sacrificing
> outcome, plus a **pluggable, per-generation-reassessed amplification layer**.

The ~0 outcome effect measured on a strong model is the EXPECTED signature of compensation against a model
that has already passed the weakness — evidence the taxonomy is correct, not that the harness failed.

---

# Scoped Repair: one unified repair backbone across all three substrates

(Added 2026-06-30.) The harness no longer "only constrains safety". Its **outcome-facing** mechanism is a
single backbone — **RepairFinding → Scoped Repair lifecycle → Delta Validation** — shared by every substrate.
A new dataset is added by writing a *repair surface adapter*, not by adding kernel branches.

## Why a backbone and not per-dataset verifiers

A channel-position study (inline self-distrust vs external reviewer assertion of the **same** finding) showed
the agent acts on harness findings far more under the external framing — but acting on a *vague, goal-level
obligation* ("write a triage note") made a weak model **overwrite already-substantive content** to look
compliant (HAB-12/-50pp, HAB-15/-60pp). Lesson: once you fix signal *position*, the binding constraint moves
to finding *content quality*. The fix must make every finding **localized, minimal, and non-degrading** —
identically for a form field, a FHIR path, or an answer claim.

## The unit: RepairFinding

`{target_type, target_path, defect_type, operation, required_change, protected_paths,
preserve_requirements, allowed_capabilities}` — names exactly ONE concrete defect, the smallest fix, and what
must be preserved. A finding with no concrete target+change is dropped (no vague REVISE). Stable `finding_id`
= hash(task, rule, target, defect) → the dedup key that stops re-nagging.

## The lifecycle (ledger.repair_findings)

`OPEN → DELIVERED → ATTEMPTED → {RESOLVED | REGRESSED | EXHAUSTED}`. A delivered finding is NOT re-sent until
the agent changes the target (anti-churn; ≤2 attempts). Every attempt is **delta-validated**.

## Delta validation (what makes it non-degrading)

`accept ⇔ target_resolved ∧ protected_content_preserved ∧ ¬new_conflict`. Structured/numeric protected
content must be EQUAL (PB dose/route can't drift); text must be retained (APPEND keeps it, OVERWRITE fails →
`repair_regression`). Reindex-tolerant for list targets. After a REMOVE, an answer-level consistency recheck
runs (a removed premise must not orphan an interpretation).

## Mapping the three repair surfaces

| Proposal surface | completeness criterion | repair operations |
|---|---|---|
| FORM (GUI portal) | required field present & sufficient | ADD / EDIT |
| FHIR (clinical API) | required path present & consistent | ADD / EDIT / REPLACE |
| ANSWER (perception) | every perceptual claim has an observation trace | REACQUIRE / VERIFY_OR_REMOVE / EDIT_OR_REMOVE |

---

# evidence_coverage: claim-conditioned observational coverage (the ANSWER gate)

The dominant failure on the perceptual substrate is **not** an incomplete write — it is the agent stating
findings it never actually observed (tool_selection / tool_path / fabrication). No prior capability touched
this; `verify_commit` is inert without a commit. `evidence_coverage` fills it, through the SAME backbone.

## Naming discipline (avoid concept-smuggling)

It is **claim-conditioned observational coverage / perceptual traceability** — NOT "gold tool-path
completeness". There is no gold path, so it can prove only:

> every *perceptual* claim in the final answer traces to an actually-executed observation of its target

It cannot (and does not) claim the agent called every tool a correct decision needs. Use `claim-observation
coverage` in code/paper; never `tool-path completeness`.

## Claims must be typed first

| claim_type | gated by evidence_coverage? |
|---|---|
| perceptual ("nodule in RLL") | YES — must have an observation trace |
| interpretive ("favors malignancy") | only requires ≥1 covered perceptual premise |
| background ("malignancy usually …") | NO |
| recommendation ("recommend biopsy") | NO |

This prevents flagging the diagnosis label itself as "no tool observed the diagnosis".

## Deterministic-first, judge at the margin

1. Ledger records a normalized observation per perception/read tool call
   `{subject, region, modality, attributes_observed, result_status}`.
2. Deterministic match: region never observed → finding; region+attribute observed → covered (no judge).
3. Judge consulted ONLY when the region was looked at but the attribute is unclear (whole-image look) —
   "do the agent's own observations support this claim?"

## Three defect cases (not always REACQUIRE)

| situation | defect_type | operation |
|---|---|---|
| target never observed | `unobserved_target` | REACQUIRE_EVIDENCE |
| observed, but observation doesn't support the claim | `unsupported_by_observation` | VERIFY_OR_REMOVE |
| claim names no concrete target | `untraceable_claim` | EDIT_OR_REMOVE |

## The judge never names a tool — the affordance registry does

The judge says only *what observation is missing*. The kernel's **affordance registry** maps that need to
*executable* tool names drawn from the task manifest (`select_tools(available_tools, region, modality)`), so
feedback can never suggest a non-existent / wrong-affordance tool. No match → no tool suggested (silent beats
hallucinated).

---

# Reclassification: layer by MECHANISM, not by file

A single capability can contain mechanisms of different layers. Classify each mechanism, not the file.

| Capability | Infrastructure part | Amplification / Compensation part |
|---|---|---|
| `repair_delta` | non-degrading delta validation (deterministic) | — |
| `evidence_coverage` | deterministic observation↔claim coverage; affordance registry | claim decomposition/classification; margin semantic support; L2 suggested-acquisition (soft only) |
| `goal_alignment` (Scoped Repair) | the lifecycle + dedup + delta | L2 semantic sufficiency (judge locates the defect) |
| `obligation_lifecycle` | obligation state machine; no-progress; dedup; retry budget | judge-inferred "which obligation is missing" |
| `scope_evidence` | — | grounding veto (Compensation; being superseded by evidence_coverage) |

A v1 caution carried from the review: L2 "a relevant tool seems unused" is **soft only** — it records an
opportunity, it does NOT block `before_final`. Blocking on "possibly-relevant tool unused" pushes a weak
agent into over-checking loops; only a *strong* observational gap (a perceptual claim with zero observation)
is allowed to REVISE.

---

# Framework (current)

```
                       ORACLE-BLIND boundary  (never reads gold / reference / checkpoints)
 ----------------------------------------------------------------------------------------
                                  AGENT  (brain: weak..strong)
       propose action -> execute -> observe -> ... -> final answer
             |                          ^                  |
   KERNEL    v before_action           | after_action     v before_final     (3 commit points)
   capabilities, layered by MECHANISM:
     INFRASTRUCTURE   subject_binding | verify_commit (+RECONCILE) | repair_delta |
                      obligation state-machine/no-progress/dedup/retry |
                      evidence_coverage: deterministic observation coverage | affordance_registry
     AMPLIFICATION    scoped_repair L2 (judge locates defect) |
                      evidence_coverage: claim decomposition / semantic support / soft suggested-acquisition |
                      obligation: judge-inferred missing obligation
     COMPENSATION     scope_evidence grounding veto (being superseded)
             |
             v  combine(ESCALATE > BLOCK > REVISE > RECONCILE > ALLOW)
     SCOPED REPAIR core (shared by all substrates):
        finding sources -> RepairFinding{target, defect, op, protected, allowed_caps}
           scoped_repair (FORM / FHIR)    evidence_coverage (ANSWER claim)
           -> ledger lifecycle: dedup -> delivered -> attempted -> DELTA-validate ->
              RESOLVED / REGRESSED(veto) / EXHAUSTED ; REMOVE -> answer-level consistency recheck
           -> repair surface adapter (the ONLY substrate-specific code): FORM | FHIR | ANSWER
     feedback render:  inline (self-distrust)  |  external (localized patch protocol;
                       tool names from affordance registry, NOT the judge)
             |
             v  back to AGENT  (ADD / EDIT / REMOVE / REACQUIRE …, preserve substantive content)

 LEDGER (harness-external state, agent cannot mutate):
   active_subject | observations{subject,region,modality,attrs,status} | evidence | obligations |
   commit_history | completed_commits | repair_findings(lifecycle) | opportunities(denominators)
```
