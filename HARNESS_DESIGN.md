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
