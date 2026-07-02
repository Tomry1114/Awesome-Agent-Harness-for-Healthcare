# Committed Workflow Completion — Unified Recovery Architecture (PB · HAB · MedCTA)

Design v2. Synthesizes a 4-agent design pass (core + 3 substrates), each grounded in the real repo
(`/hpc2hdd/home/ce483/Medical_harness/`) and real runs. Supersedes the "COMPLETE-effect = complete the last
mechanical action" model. No production code here — this is the design others implement.

## 0. Thesis and the correction

**Wrong boundary (previous):** recovery may only complete the *last single mechanical action* (one `fhir_create`,
one `click`). This conflated two different things — "the harness must not make a **new decision**" with "the harness
may execute only **one step**" — and bent `EffectPlan`, the adapters, and the orchestrator around a single write.

**Correct boundary: INFORMATION SUFFICIENCY, not step count.** When an agent has committed a goal whose remaining
execution is fully determined by already-decided values, the harness may run the **entire deterministic execution
closure** of that goal — multiple reads, navigations, reversible edits, and one-or-more irreversible commits —
stopping *only* when a step needs a **new decision, new content, or an unknown parameter**. Multi-step is not
overreach; an **unbound argument** is.

**Rename:** COMPLETE-effect → **Committed Workflow Completion (CWC)** = Plan Completion (deterministic execution of
a committed plan) + Effect Verification. No benchmark names in core; substrates are *structured-record*,
*interactive-GUI*, *perceptual*.

**Preserved invariants (unchanged):** oracle-blind; Evidence Ledger; the auth machine
`AVAILABLE→RESERVED→DISPATCHED→VERIFIED/UNKNOWN/FAILED/CANCELLED`; exact auth-id binding; single `ActionExecutor`
pipeline; before/after_action; bounded execution; adapter seam; controlled-replay attribution
(`action_id`/`recovery_episode_id`); root-agent provenance; **no repeat mutation after UNKNOWN**. What changes:
the *shape* of the plan (one write → bounded step list) and authorization (one irreversible tier → three tiers).

## 1. Core pipeline (six components)

1. **Gap Router** — classify the failure: `evidence | decision | execution | verification`. Only *execution*
   gaps are completed; *decision* gaps are terminal `BLOCKED_NEEDS_DECISION` (the harness must not decide);
   *evidence* gaps are recoverable only as read-only prerequisites of an execution gap; *verification* gaps are
   resolved by idempotent re-read. Maps onto existing `gap.py` (MISSING_CONTEXT/UNOBSERVED_FEATURE→evidence,
   INTERPRETATION→decision, INCOMPLETE_EFFECT→execution, UNRESOLVED_POSTCONDITION→verification).
2. **Commitment Resolver** — oracle-blind: from root deliverable + trajectory + authoritative state (never
   gold/checkpoint), produce ONE `CommittedGoal` carrying the agent's frozen `committed_decisions`.
3. **Workflow Compiler** — compile the `CommittedGoal` → a **bounded, ordered `EffectPlan`** via the substrate adapter.
4. **Decision Boundary Checker** — per step, prove every argument traces to a legal source; else the step (and
   episode) BLOCKS. This is the information-sufficiency boundary made operational.
5. **Unified ActionExecutor** — read / navigate / reversible_write / irreversible_commit all run through ONE
   auth+execute+after_action chain (the existing strict pipeline).
6. **Verification** — did the target effect land (server read-back / state-marker flip / evidence-coverage recheck).

## 2. Core dataclasses (v2)

```
ArgumentBinding(name, value, source ∈ {agent_commitment, authoritative_state, unique_workflow_structure,
                bound_evidence}, provenance)
   # INVARIANT: constructed ONLY when a real source is found. No default/guessed source; harness-generated
   # values forbidden. An argument with no ArgumentBinding is UNBOUND → the step and episode BLOCK.

CommittedGoal(goal_id, intent, target_entity, committed_decisions[], effect_type, signature, origin_action_ids[])
   # committed_decisions = the FROZEN agent decisions; the ONLY values a step may treat as agent_commitment.

RecoveryStep(kind ∈ read|navigate|reversible_write|irreversible_commit|verify, action, arg_bindings{},
             affordance{tool,match:{labels,role},target_key}, expected_postcondition, idempotency_key, reversible)

EffectPlan(goal, steps[], required_bindings[], stop_conditions[], expected_postcondition, scope_of(step)->scope)
   # REPLACES the single-mutation EffectPlan. steps = the deterministic execution closure of `goal`.

EpisodeResult(state, goal_id, completed_steps[], blocked_step_index, blocked_argument, created_ids[],
              auth_status, reason, prereq_rounds, events)   # EXTENDS today's; realized = state ∈ {VERIFIED, ALREADY_REALIZED}
```

The four legal `source`s (only these close the boundary): **agent_commitment** (a value in
`committed_decisions`); **authoritative_state** (read back via the strict reader); **unique_workflow_structure**
(the substrate exposes exactly ONE matching affordance/invariant; non-unique → `BLOCKED_AMBIGUOUS_TARGET`);
**bound_evidence** (a ledger record in PRESENT/ABSENT, subject-bound, from ACQUIRING). `EffectCompletionKey`
(subject, artifact_hash, signature, effect_type) is kept for episode dedup.

## 3. Orchestrator v2 — state machine + tiered authorization

States: `NOT_STARTED → ACQUIRING → PLANNING → EXECUTING_REVERSIBLE → READY_TO_COMMIT → COMMITTING → VERIFYING →
VERIFIED`. Terminals: `BLOCKED_NEEDS_DECISION`, `BLOCKED_MISSING_EVIDENCE`, `BLOCKED_AMBIGUOUS_TARGET`, `FAILED`,
`UNKNOWN`, plus dedup short-circuit `ALREADY_REALIZED`. (Splits the old `BLOCKED_TERMINAL` into the three
`BLOCKED_*` — so "correctly refused because a decision was needed" is distinguishable from a failure; old
`RECONCILING` == new `UNKNOWN`.)

Per step, two mandatory gates before execution: **(a) Decision Boundary** — for each arg, `adapter.resolve_binding`
returns an `ArgumentBinding` or `None`→BLOCK immediately; **(b) Tiered authorization**:

| step kind | tier | authorization | executor path | verify |
|---|---|---|---|---|
| read, navigate | read/nav | none minted; strict read gate (before_action ALLOW) | `execute_recovery_read` (P0-1/2/3) | ledger evidence delta PRESENT/ABSENT |
| reversible_write | reversible | scoped `MutationAuthorization(tier=reversible, effect=reversible)`, exact-scope, single-use; failed edit may bounded-retry | `execute_reversible` (new; generalizes `_execute_gui_marker`) | state read-back changed |
| irreversible_commit | irreversible | exact `MutationAuthorization(tier=irreversible)` + auth-id bound + idempotency_key; **no re-commit after UNKNOWN** | `execute` (existing) | server/state read-back |

`MutationAuthorization` gains ONE field (`tier`); the `AVAILABLE→…→CANCELLED` machine and `exact_scope_match` are
unchanged. `mint_authorization` forwards `tier`. Read steps mint nothing (they already run the strict reader).
Both reversible and irreversible require an exact `target_path` + a verifiable `expected_postcondition`
(the existing deterministic-gap guard) — no blanket "fix the form" authority.

**Control flow (per episode):** classify_gap → (decision → BLOCK) → resolve_commitment (None → FAILED) → dedup key
→ plan-level inspect (PRESENT → ALREADY_REALIZED) → ACQUIRING (resolve bound_evidence prereqs, bounded; UNKNOWN →
BLOCKED_MISSING_EVIDENCE) → PLANNING (compile; empty → FAILED) → walk steps [boundary-check each arg → tiered
execute; unbound → BLOCKED_*; reversible/commit UNKNOWN → UNKNOWN reconcile-only; commit FAILED → bounded retry] →
VERIFYING (plan postcondition True→VERIFIED / None→UNKNOWN / False→FAILED). Every step still: one ActionExecutor
pass, scoped exact-id single-use auth at current evidence version, hard budget gate before each env call,
`action_id`/`episode_id` stamped, `EffectCompletionKey` dedup.

## 4. RecoveryAdapter interface v2

```
should_trigger(lifecycle_event) -> bool          # PB=deliverable_confirmed; GUI/perceptual=before_final
context(task) -> dict
classify_gap(ctx, ledger, trajectory) -> GapSignal
resolve_commitment(root, trajectory, goal, judge, ctx) -> CommittedGoal | None   # was extract_commitments
goal_key(goal, ctx) -> EffectCompletionKey                                        # was effect_key
compile_workflow(goal, ctx, manifest) -> EffectPlan | None                        # was compile_effect (multi-step now)
resolve_binding(arg_name, step, goal, ctx, ledger) -> ArgumentBinding | None      # NEW — the boundary hook
resolve_affordance(step, state_view) -> dict | None                               # generalizes resolve_document_affordance
inspect_effect(goal, step, driver, ctx) -> EffectInspection
verify_postcondition(expected, state_view) -> bool | None
```

`FhirRecoveryAdapter` / `GuiRecoveryAdapter` become v2 implementations; a new `ToolSandboxRecoveryAdapter`
(MedCTA) fills the current `None`. `compile_evidence_plan` (the base method, currently returns None) is the
MedCTA read-only ACQUIRE compiler.

## 5. Substrate mappings

### 5.1 PB — structured-record (FHIR)  [proven: cp3 pelvic-US, created_id 212151, VERIFIED]

- **Gap:** execution = agent wrote an unconditional order in the deliverable but never `fhir_create`d (recoverable).
  evidence = a governance read never issued (recoverable only as prereq). decision = wrong/absent clinical
  judgment, e.g. cp2_clinical_reasoning / cp4_pregnancy_test_decision (NOT recoverable). verification = create
  landed but read-back ambiguous (idempotent re-read).
- **Commitment:** `extract_committed_orders` (judge, oracle-blind) keeps only firm imperative orders; drops
  hedged/conditional → decision-gap cases never form a commitment. Field bindings: `code.text ← agent_commitment`;
  `subject/authoredOn/requester ← authoritative_state` (public context); `status/intent/resourceType ←
  unique_workflow_structure` (category→resourceType map).
- **Plan (4 steps):** S1 governance read (`fhir_search Allergy/MedicationRequest`; UNKNOWN→BLOCKED_MISSING_EVIDENCE;
  no-op for imaging) → S2 existing-effect probe (`fhir_search ServiceRequest`; PRESENT→NO-OP, UNKNOWN→BLOCK never
  create) → S3 irreversible_commit (`fhir_create`, exact auth, tag `harness-recovery-created`) → S4 verify
  (`reconcile_write` GET; landed=server-confirmed).
- **Decision boundary:** ambiguous drug/dose (`start metformin`, no dose → dosage args untraceable) →
  BLOCKED_NEEDS_DECISION; allergy conflict → never substitute → BLOCKED_NEEDS_DECISION; multiple MRNs →
  BLOCKED_AMBIGUOUS_TARGET.
- **Auth tiers:** read = `fhir_search`/`fhir_read`; irreversible = `fhir_*_create` (medication path = highest
  scrutiny + mandatory S1).

### 5.2 HAB — interactive-GUI (real Playwright portal, 127.0.0.1:3002)

- **Gap (grounded in the real HAB-denial-medium-1 run):** the weak agent REACHED the dispute form (typed member
  id → Search → View Details → opened form) but the form required a **medical-necessity rationale** (net-new
  content) + a **downloaded attachment** (evidence) it never prepared → a **decision/evidence gap → BLOCK**, not a
  clean execution gap. Contrast denial_triage (res6): agent selected disposition, skipped `documentedAppealInEpic`
  → clean execution gap (works).
- **Commitment present only if:** goal binds the case; a disposition/decision landed in authoritative state; the
  rationale CONTENT was authored by the agent; the attachment was downloaded; `submittedAppeal` still False.
  Otherwise a PARTIAL commitment covering only mechanical prefix, then BLOCK.
- **Plan (electronic appeal, ~12 steps):** snapshot → navigate payer portal → type member_id (agent_commitment) →
  Search → View Details (row bound by claim_id, else BLOCKED_AMBIGUOUS_TARGET) → open Dispute (unique_workflow_
  structure) → fill Contact (authoritative_state) → fill Rationale (**bound_evidence; BLOCKED_MISSING_EVIDENCE if
  unauthored**) → upload attachment (**bound_evidence; BLOCKED_MISSING_EVIDENCE if not downloaded**) →
  irreversible_commit Submit Appeal → verify `appealActions.submittedAppeal False→True` → document confirmation
  number (authoritative_state, only AFTER submit returns it).
- **Real H2 (affordance resolution, never built):** refs are re-numbered every snapshot, so a step carries a
  label/role matcher, resolved against the FRESHEST snapshot immediately before acting. `resolve_affordance`: parse
  interactive-element lines → filter by role → normalize+match labels → require exactly ONE survivor (0 →
  BLOCKED_MISSING_TARGET usually an ordering violation; ≥2 → BLOCKED_AMBIGUOUS_TARGET unless a bound id
  disambiguates); a resolved-but-ineffective click (validation-gated form; NoProgress `state_changed=False`) →
  BLOCK, not success. Handle two-stage dropdowns (button→option).
- **Substrate fix (required):** `GuiEnvReal._read_state()` reads only `portals_state.emr`; extend it to also read
  `payer_a_state`/`payer_b_state` — otherwise every `payer_a_state.*` checkpoint resolves to None (why the real
  run's cp4–cp7 are `got=None`, not `False`).
- **Auth tiers:** navigate/search = read gate; type/fill/upload = reversible; submit/send_appeal = irreversible.

### 5.3 MedCTA — perceptual (image-tool sandbox)  [irreversible-mutation branch OFF]

- **Gap:** dominant = EVIDENCE gap (answered without looking) → MISSING_CONTEXT/UNOBSERVED_FEATURE → ACQUIRE. NOT
  recoverable = decision gap (looked but reasoned wrong → re-reading is a no-progress loop). Distinguished by
  `region_observed(...) != None` (whole-image counts as weak coverage) and the discriminator returning `region:null`.
- **Commitment = AnswerSlot A** `{finding, diagnosis, answer_choice, confidence}` from the agent's OWN final answer
  (`decompose_claims`/`extract_decision_signature`), never `reference.gold_answer`. Missing = perceptual claims
  failing `region_observed`/`attribute_observed`. Forms only if: a final answer exists; an enforceable
  `unobserved_target` finding exists; the missing observation has a uniquely resolvable perception affordance.
- **Plan (read-only ACQUIRE via `compile_evidence_plan`):** whole-image `ImageDescription` / `RegionAttribute
  Description` (region ← PUBLIC QUESTION only, via `elicit_discriminator`; NEVER gold bbox) / OCR (only if question
  cites text). No `mutation_action`, no `resource`.
- **"Commit" = internal answer-slot revision A→B**, governed by `evaluate_candidate` (AnswerRetention): ADOPT_B
  only if new validated evidence directly supports the change at conf≥0.8, comparator clearly prefers B, no new
  unsupported claim, non-target content preserved, B re-passes the coverage gate; else KEEP_A (T→F harm control).
- **Decision boundary:** region not uniquely derivable from the question → BLOCKED_AMBIGUOUS_TARGET;
  `localization.resolved==False` → BLOCKED_MISSING_EVIDENCE; empty OCR / no-result web → no acquired evidence.
- **Evidence-coverage gate:** every visual claim must bind to `source_channel==radiology_image` evidence; a claim
  supported only by `GoogleSearch` (`external_web`) is rejected (web text corroborates background, never grounds an
  image finding — manifest declares the channel split). Catches: answered-without-looking, whole-image-claims-
  detail, region-failed-still-claims, empty-OCR-cited.
- **Core reuse:** Gap Router, Commitment Resolver, Workflow Compiler, Decision Boundary, ActionExecutor+
  EvidenceState (ACQUIRE reads), controlled replay (tool_sandbox replay mode), provenance — YES. MutationAuthorization
  irreversible commit + server read-back — NO (no environment write). Verification = coverage(B) clean on image
  channel AND evaluate_candidate==ADOPT_B.

## 6. Decision-boundary rules (unified)

| condition | terminal |
|---|---|
| an unconditional agent decision is missing / hedged / not landed | (no commitment) — never fires |
| a required argument has no legal source | BLOCKED_NEEDS_DECISION |
| net-new content must be authored (rationale, clinical indication, dose) | BLOCKED_NEEDS_DECISION |
| required evidence unresolved (governance read UNKNOWN; attachment not downloaded; localization.resolved False) | BLOCKED_MISSING_EVIDENCE |
| target/affordance not uniquely resolvable (>1 claim, >1 button, region ambiguous) | BLOCKED_AMBIGUOUS_TARGET |
| commit dispatched but read-back ambiguous | UNKNOWN (reconcile-only, no re-commit) |
| effect already present | ALREADY_REALIZED (no-op) |

## 7. Refactor map (keep / extend / replace)

REPLACE: `RecoveryAdapter` (7 methods → v2), `EffectPlan` (single mutation → steps[]), `RecoveryOrchestrator._run`
(single write → multi-step walk). EXTEND: `Commitment`→`CommittedGoal.committed_decisions`; `EpisodeResult`
(+completed_steps/blocked_*); orchestrator states (split BLOCKED_TERMINAL, RECONCILING→UNKNOWN); `RunDriver`
(+execute_reversible/navigate, promote resolve_affordance); `MutationAuthorization` (+tier); `mint_authorization`
(+tier); `gap.py` (+4-way class); `effect_completion.py`/`plan_completeness.py` (→ substrate compile_workflow
bodies). KEEP verbatim: strict `execute_recovery_read` (read tier), `ActionExecutor`, auth machine +
`exact_scope_match` + deterministic-gap guard, `EffectCompletionKey`, budget gate, `action_id`/episode stamping,
EvidenceState/adapter_compiler, provenance, no-repeat-after-UNKNOWN. NEW code: GUI DOM-ref `resolve_affordance`
(`affordance.py` is perception-only today); `GuiEnvReal._read_state` payer-state extension;
`ToolSandboxRecoveryAdapter`.

## 8. Honest applicability per dataset (where a real recoverable population exists)

- **PB (FHIR):** REAL and proven — agent commits an order in text, forgets the tool call. Clean execution gap;
  cp3 flip is reproduced and attributable. Strongest COMPLETE-effect substrate.
- **HAB (GUI):** free-text electronic-appeals under weak agents → recovery correctly BLOCKS (rationale/attachment
  are un-fabricatable) — a near-empty recoverable population TODAY. Real population is concentrated in
  **(a) decision-documentation** (select disposition, skip document-in-Epic — already works) and **(b) submit_auth
  structured-form completion** (all fields from authoritative EMR state; agent gathered codes, stalled before
  submit — a clean multi-step execution gap). Population grows with agent strength (mid-strength agents that
  prepare content then stall on mechanics).
- **MedCTA (perceptual):** REAL evidence-gap population (answered-without-looking); recovery = ACQUIRE + guarded
  answer revision; no mutation. Reuses coverage + AnswerRetention.

## 9. Implementation order

1. **Core v2 primitives** (no behavior change): dataclasses (`CommittedGoal/RecoveryStep/ArgumentBinding/
   EffectPlan-v2/EpisodeResult-v2`), orchestrator states + per-step boundary+tiered-auth walk, `MutationAuthorization.tier`,
   `resolve_binding`/`compile_workflow` adapter hooks. Regression gate: PB cp3 still F→T (FhirRecoveryAdapter re-expressed
   as a 1-commit plan, byte-identical).
2. **PB adapter v2** = 4-step plan (S1..S4) with typed ArgumentBindings + BLOCKED_* terminals. Extends to
   medication/lab/referral; refuses decision gaps at extraction.
3. **HAB substrate prerequisites**: extend `GuiEnvReal._read_state` (payer_a/b_state); build the real
   `resolve_affordance` (live-obs DOM-ref resolver). Then HAB adapter v2 = the multi-step appeal/submit_auth plan
   with the rationale/attachment BLOCK boundary. Target the recoverable population: submit_auth structured forms +
   decision-documentation (NOT free-text appeals for weak agents).
4. **MedCTA adapter** = `ToolSandboxRecoveryAdapter` (compile_evidence_plan + AnswerRetention commit + coverage
   gate); no mutation machinery. First gate: answered-without-image → force one ImageDescription → one grounded
   revision.
5. **Attribution**: controlled-replay per substrate (record agent trace → replay OFF/ON, only harness differs);
   report per-dataset outcome gain + BLOCK-rate + avg recovery cost + zero T→F. Report BLOCK as a first-class
   correct outcome, not a failure.
