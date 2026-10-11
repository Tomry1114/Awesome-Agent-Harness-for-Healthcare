"""PhysicianBench default-workspace middleware: the deliverable lifecycle.

The note the task asks for, from first write to final answer: a low-budget warning while it is missing, a nudge
before the final if it was never written, freshness / specificity / commitment-ledger nudges on the written
note, a retention check that keeps a rewrite from dropping what the previous version had, and one forced write
at the end of the episode. The hook methods at the bottom are how the shell (runner/middleware_host.py) calls it.

Inert unless the task names an output/ deliverable path. It reads the note through the shell's read-only view of the
agent's output directory (ctx.output_files()), the trajectory through ctx.trajectory, and records through ctx.emit
(and ctx.record_budget_warning for the one reserved record it asks for).
"""
import posixpath

from workspace_api import Observation, Revision

ENV_TYPE = "fhir"   # this workspace's substrate (manifest.yaml env_type): whose tool schemas declare the arguments


class DeliverableMiddleware(object):
    # PB-specific deliverable nudging/enforcement, kept out of the generic runner loop so that loop stays
    # benchmark-agnostic. For tasks without a deliverable path .active is False and every method no-ops.
    def __init__(self, task):
        import re
        # active=bool(self.path) gates every method below.
        _pt = str(task.get("goal") or "") + "\n" + str(task.get("context") or "")   # PB tasks name the deliverable path in context, not goal, so both are searched
        m = re.search(r"(?:/?workspace/)?output/[\w.\-]+", _pt)
        self.path = m.group(0) if m else None
        self.nudges = 0
        self.stale_nudges = 0
        self.ledger_nudges = 0
        self.spec_nudges = 0
        self._frozen_commits = None   # frozen execute_now/asserted_done commitments (append-only union across rounds)
        self._judge_errors = []       # judge-call failures (403/timeout/etc): recorded so a JUDGE OUTAGE is
                                      # distinguishable from "no opportunity" -- run.py drains this into
                                      # result["harness"]["runtime_errors"] (never silently swallowed).
        self.budget_nudged = False
        self._retain_A = None       # snapshot of on-disk deliverable version A held across a nudged rewrite
        self.retention_blocks = 0   # bounded count of regressive-rewrite rejections (fail-open at cap)
        self._declared_path_args = {}   # tool -> the parameters it declares for the file it writes (_path_args)

    @property
    def active(self):
        return bool(self.path)

    def is_required_write(self, action):
        # True ONLY for the exact required deliverable write (right tool + right filename) so the over-budget
        # exception covers one precise action, not any write_file. Keeps the runner from naming a raw tool.
        # The filename is read from the argument the tool declares for it (_path_args), not from a guessed name.
        if not self.active or (action or {}).get("tool") != "write_file":
            return False
        args = action.get("args") or {}
        named = [a for a in self._path_args(action.get("tool")) if a in args]
        if not named:
            return False
        supplied = posixpath.normpath(str(args.get(named[0]) or "")).lstrip("/")
        required = posixpath.normpath(str(self.path or "")).lstrip("/")
        return bool(required) and posixpath.basename(supplied) == posixpath.basename(required)

    def _path_args(self, tool):
        """The parameters through which `tool` names the file it writes, as the tool declares them: the parameters
        of its published schema (harness.substrate_catalog.tool_parameters, i.e. the upstream PhysicianBench
        registry PbNativeEnv dispatches) called `path` or `*_path`. For write_file that is file_path. When the
        schemas cannot be read the answer is none -- no write is taken for the deliverable -- and the error is
        reported through deferred_errors."""
        if tool not in self._declared_path_args:
            try:
                from workspace_api import tool_parameters
                params = (tool_parameters(ENV_TYPE) or {}).get(tool) or []
            except Exception as e:
                params = []
                self._judge_errors.append("deliverable_path_args: %r" % (e,))
            self._declared_path_args[tool] = [p for p in params if p == "path" or p.endswith("_path")]
        return self._declared_path_args[tool]

    def _file(self, ctx):
        """The deliverable's entry {size, text} in the agent's output directory, or None (no output directory, or no
        such regular file)."""
        out = ctx.output_files()
        if out is None:
            return None
        return (out.get("files") or {}).get(posixpath.basename(self.path))

    def _want(self, ctx):
        """The deliverable's file name when the environment has an output directory, else ""."""
        return posixpath.basename(self.path) if ctx.output_files() is not None else ""

    def _missing(self, ctx):
        f = self._file(ctx)
        return not (f and f.get("size", 0) > 0)

    def budget_warning(self, ctx):
        # one-shot low-budget warning if the deliverable is still missing; keyed on EXECUTED env actions
        # (not loop steps) so harness repair turns don't make PB think the task budget is nearly spent.
        step, max_steps = ctx.used_actions, ctx.max_steps
        if not self.active or self.budget_nudged or (max_steps - step) > 8:
            return None
        if not self._want(ctx) or not self._missing(ctx):
            return None
        self.budget_nudged = True
        ctx.record_budget_warning()     # the shell's reserved record: step of this context, remaining = max - used
        fb = ("You are RUNNING OUT OF STEPS (only %d left) and have NOT written the required deliverable. "
              "If you have already gathered the data the task requires, stop exploring and immediately call "
              "write_file with EXACTLY path=\"%s\" and content = your full clinical assessment and "
              "management plan. If a data category the task explicitly requires is still unqueried, complete "
              "that one read FIRST, then write the file." % (max_steps - step, self.path))
        return {"feedback": fb}, "budget_warning"

    @staticmethod
    def _confirmed_order_displays(trajectory):
        # display strings of every landed create whose server read-back was CONFIRMED (reconciliation
        # confirmed=True), using the same verified-receipt rule as the freshness check below, incl. the has_recon
        # fallback (raw id counts ONLY when no create reconciliation ran at all), so the order checks never
        # contradict each other. Oracle-blind: every string is the AGENT's OWN order text (its args / the server
        # echo), never a hidden checkpoint value.
        import re as _re
        confirmed = set(); has_recon = False
        for e in (trajectory or []):
            if e.get("event_type") == "reconciliation" and str(e.get("tool", "")).endswith("_create"):
                has_recon = True
                if e.get("confirmed") is True:
                    m = _re.search(r"([A-Za-z]+)/([\w.-]+)", str(e.get("detail") or ""))
                    if m: confirmed.add((m.group(1), m.group(2)))
        out = []
        for e in (trajectory or []):
            if e.get("event_type") != "tool_call":
                continue
            tool = str(e.get("tool", "")); r = e.get("result")
            if not (tool.endswith("_create") and isinstance(r, dict) and r.get("id")
                    and r.get("resourceType") != "OperationOutcome"):
                continue
            if has_recon and (str(r.get("resourceType") or ""), str(r.get("id"))) not in confirmed:
                continue
            a = e.get("args") or {}
            parts = [str(a.get(k) or "") for k in ("code_display", "code", "code_code",
                                                   "medication_display", "reason_display", "category_display")]
            for _blk in (r.get("code"), r.get("medicationCodeableConcept")):
                if isinstance(_blk, dict):
                    parts.append(str(_blk.get("text") or ""))
                    for _cod in (_blk.get("coding") or []):
                        if isinstance(_cod, dict):
                            parts.append(str(_cod.get("display") or ""))
            disp = " ".join(p for p in parts if p).strip()
            if disp:
                out.append(disp)
        return out

    _COMMITMENT_PROMPT = (
        "You extract the clinician's OWN commitments from THEIR final note -- nothing else. Read the NOTE below "
        "(written by the clinician) and list the SPECIFIC orders they present as being placed NOW / already "
        "placed. Copy the clinician's OWN wording for the specific order name only (a named test, study, "
        "medication, or referral target). Two buckets:\n"
        "  execute_now  = stated as an action to take now: 'order a 12-lead ECG', 'obtain a TSH', "
        "'refer to electrophysiology', 'place a plasma metanephrines order'.\n"
        "  asserted_done = stated as ALREADY completed: 'a 12-lead ECG has been ordered', 'TSH was sent'.\n"
        "Do NOT include anything the clinician only RECOMMENDS / would CONSIDER / says to do IF a condition "
        "holds ('consider', 'recommend', 'if ...') -- those are not commitments. Do NOT include vague umbrella "
        "phrases ('workup', 'labs as indicated') -- only specifically named orders. Invent nothing not in the "
        "note. Reply STRICT JSON {{\"execute_now\": [\"<order name>\", ..], \"asserted_done\": "
        "[\"<order name>\", ..]}} (empty lists if none).\n\nNOTE:\n{note}\n")

    def deliverable_freshness_nudge(self, ctx, trajectory, judge):
        # The deliverable must reflect the FINAL orders/evidence. If the agent landed a create AFTER
        # the last write_file, the file is STALE (documents e.g. "no orders" while orders now exist) -> the
        # documentation/reasoning checkpoints read a contradictory file. Re-enter to update it. Structural
        # (write step vs later effect), oracle-blind, no content authored. Capped at 2 nudges.
        if not self.active or self.stale_nudges >= 2 or self._missing(ctx):
            return None
        payload = self._detect_freshness_payload(trajectory)
        if payload is None:
            return None
        return self._deliver_freshness(payload, ctx)

    def _detect_freshness_payload(self, trajectory):
        # pure structural detection: returns (n_later_effects, n_new_evidence) or None (no staleness / no write).
        wsteps = [e.get("step", 0) for e in (trajectory or [])
                  if e.get("event_type") == "tool_call" and e.get("tool") == "write_file"]
        if not wsteps:
            return None
        last_wf = max(wsteps)
        # only READ-BACK-CONFIRMED effects count as a later effect -- the same verified-receipt rule the
        # commitment ledger uses, so the checks never give contradictory feedback (one saying the order is not
        # confirmed while the other says an order was placed).
        import re as _re
        confirmed = set(); has_recon = False
        for e in (trajectory or []):
            if e.get("event_type") == "reconciliation" and str(e.get("tool", "")).endswith("_create"):
                has_recon = True
                if e.get("confirmed") is True:
                    m = _re.search(r"([A-Za-z]+)/([\w.-]+)", str(e.get("detail") or ""))
                    if m: confirmed.add((m.group(1), m.group(2)))
        later = [e for e in (trajectory or [])
                 if e.get("event_type") == "tool_call" and str(e.get("tool", "")).endswith("_create")
                 and e.get("tool") != "write_file" and isinstance(e.get("result"), dict)
                 and e["result"].get("id") and e["result"].get("resourceType") != "OperationOutcome"
                 and e.get("step", 0) > last_wf
                 and (not has_recon or (str(e["result"].get("resourceType") or ""), str(e["result"].get("id"))) in confirmed)]
        # NEW EVIDENCE after the last write is staleness -- but ONLY genuinely NEW evidence, not a
        # repeat query of data the file already reflects. Signature each retrieved resource as
        # (resourceType, id, versionId, effectiveDate). Evidence retrieved BEFORE/AT the last write is
        # assumed reflected; a post-write search is stale-making ONLY if it surfaces a signature not seen
        # pre-write (a new or updated resource), never a re-query of already-retrieved data. Structural;
        # oracle-blind (identity/version/date only, never a value or code).
        def _ev_sig(ent):
            rt = str(ent.get("resourceType") or "")
            rid = str(ent.get("id") or "")
            if not (rt and rid):
                return None
            ver = str((ent.get("meta") or {}).get("versionId") or "")
            eff = (ent.get("effectiveDateTime") or ent.get("issued") or ent.get("authoredOn")
                   or ent.get("recordedDate") or ent.get("occurrenceDateTime")
                   or ent.get("performedDateTime") or ent.get("date")
                   or (ent.get("effectivePeriod") or {}).get("start")
                   or (ent.get("period") or {}).get("start") or "")
            return (rt, rid, ver, str(eff))
        # the agent's OWN landed creates are handled by the `later` branch above; exclude them from the
        # evidence diff so reading back a just-placed order (already reflected in the file) is NOT counted
        # as "new patient data" (avoids a double-fire / false-stale rewrite).
        created_ids = set()
        for e in (trajectory or []):
            if e.get("event_type") == "tool_call" and str(e.get("tool", "")).endswith("_create"):
                rr = e.get("result")
                if isinstance(rr, dict) and rr.get("id") and rr.get("resourceType") != "OperationOutcome":
                    created_ids.add((str(rr.get("resourceType") or ""), str(rr.get("id"))))
        pre_sigs, post_sigs = set(), set()
        for e in (trajectory or []):
            if e.get("event_type") != "tool_call" or "search" not in str(e.get("tool", "")):
                continue
            r = e.get("result")
            if not (isinstance(r, dict) and r.get("entries")
                    and str(r.get("resourceType")) != "OperationOutcome"):
                continue
            bucket = post_sigs if e.get("step", 0) > last_wf else pre_sigs
            for ent in (r.get("entries") or []):
                if not isinstance(ent, dict):
                    continue
                if (str(ent.get("resourceType") or ""), str(ent.get("id") or "")) in created_ids:
                    continue
                s = _ev_sig(ent)
                if s is not None:
                    bucket.add(s)
        new_ev = post_sigs - pre_sigs   # signatures the file could NOT already reflect
        if not later and not new_ev:
            return None
        return (len(later), len(new_ev))

    def _deliver_freshness(self, payload, ctx):
        # delivery half: increment + build fb + log. `payload` = (n_later, n_new).
        n_later, n_new = payload
        self.stale_nudges += 1
        _bits = []
        if n_later:
            _bits.append("placed %d order(s)" % n_later)
        if n_new:
            _bits.append("retrieved new patient data")
        fb = ("You %s AFTER you last wrote your deliverable, so the file no longer matches the patient's chart. "
              "Update the deliverable file now to reflect the orders you placed and all evidence you have "
              "gathered, then finish." % " and ".join(_bits))
        ctx.emit({"step": ctx.step, "event_type": "deliverable_stale_nudge",
                  "n_later_effects": n_later, "n_later_evidence": n_new,
                  "attempt": self.stale_nudges, "status": "ok"})
        return {"feedback": fb}, fb

    def arm_retention(self, ctx):
        # Artifact retention: snapshot the CURRENT on-disk deliverable (version A) at the moment a
        # correction nudge sends the agent back to rewrite it, so the follow-up full-'w' write_file can be
        # checked for REGRESSION before it lands. No-op when no deliverable exists yet (nothing to protect ->
        # the FIRST write is never gated). Idempotent: always re-captures the latest ACCEPTED version, so
        # successive nudges each protect the newest good file.
        if not self.active or self._missing(ctx):
            self._retain_A = None
            return
        try:
            self._retain_A = self._file(ctx)["text"]
        except Exception:
            self._retain_A = None

    def _read_deliverable(self, ctx):
        # the deliverable's text; without it, the largest regular file of the output directory (first in listing
        # order on a tie); "" without an output directory or a file
        out = ctx.output_files()
        if out is None or not out.get("exists"):
            return ""
        files = out.get("files") or {}
        cand = []
        if self.path and posixpath.basename(self.path) in files:
            cand = [posixpath.basename(self.path)]
        if not cand:
            cand = list(files)
        if not cand:
            return ""
        try:
            return files[max(cand, key=lambda name: files[name]["size"])]["text"]
        except Exception:
            return ""

    @staticmethod
    def _retrieved_summary(trajectory):
        # List the DISPLAY NAMES of every resource the agent retrieved rather than truncated raw JSON:
        # truncation can hide items (e.g. TSH, prolactin) from the judge and produce false 'missing'. The judge
        # needs the full NAME list to tell presence; values/payloads are irrelevant to a presence check.
        names, seen = [], set()
        for e in (trajectory or []):
            if e.get("event_type") != "tool_call" or "search" not in str(e.get("tool", "")):
                continue
            r = e.get("result")
            if not (isinstance(r, dict) and r.get("entries")):
                continue
            for ent in r.get("entries") or []:
                if not isinstance(ent, dict):
                    continue
                cc = ent.get("code") or {}
                nm = cc.get("text") or ""
                if not nm:
                    for cod in (cc.get("coding") or []):
                        if isinstance(cod, dict) and cod.get("display"):
                            nm = cod["display"]; break
                if not nm:
                    md = ent.get("medicationCodeableConcept") or {}
                    nm = md.get("text") or ""
                    if not nm:
                        for cod in (md.get("coding") or []):
                            if isinstance(cod, dict) and cod.get("display"):
                                nm = cod["display"]; break
                if not nm:
                    nm = ent.get("display") or (ent.get("resourceType") or "")
                nm = str(nm).strip()
                k = nm.lower()
                if nm and k not in seen:
                    seen.add(k); names.append(nm)
                if len(names) >= 400:
                    break
        if not names:
            return ""
        return "Data items the clinician has already retrieved:\n" + "; ".join(names[:400])

    _MATCH_PROMPT = (
        "A clinician committed in their plan to these specific actions:\n{commits}\n\n"
        "These orders were actually placed in the chart:\n{orders}\n\n"
        "For EACH committed action, decide whether one of the placed orders FULFILLS it -- the SAME clinical "
        "action, allowing clinical synonyms/abbreviations (e.g. TSH = Thyrotropin, ECG = EKG = electrocardiogram, "
        "echo / TTE = transthoracic echocardiography, EP = electrophysiology). Each placed order fulfills AT MOST "
        "ONE commitment. Return the committed actions that are NOT fulfilled by any placed order, copied VERBATIM "
        "from the committed-actions list. Reply STRICT JSON {{\"unfulfilled\": [\"<committed action verbatim>\", ..]}} "
        "(empty list if every commitment is fulfilled)."
    )

    _SPEC_COMPLETION_PROMPT = (
        "You check the COMPLETENESS of a clinician's note against the PUBLIC task -- never clinical correctness, "
        "and you NEVER supply an answer or value. Inputs: the PUBLIC instruction, the data the clinician "
        "RETRIEVED (names/values), and the clinician's NOTE. List the items the PUBLIC instruction REQUIRES that "
        "the NOTE does not CONCRETELY provide -- either (a) a required dimension left vague or absent (e.g. the "
        "instruction requires a dosing + titration schedule but the note only says 'consider pharmacotherapy'; "
        "or requires an explicit order/defer decision but the note is non-committal; or requires a specific "
        "section the note omits), or (b) a specific datum the instruction says to REPORT that the clinician "
        "RETRIEVED but did not state in the note. For EACH, describe WHAT THE PUBLIC TASK REQUIRES (the slot), "
        "NOT the value/answer. Do NOT invent a requirement that is not in the public instruction. Reply STRICT "
        "JSON {{\"gaps\": [\"<what the public task requires that is missing or vague>\", ..]}} (empty list if the "
        "note concretely and completely addresses every public requirement).\n\n"
        "PUBLIC INSTRUCTION:\n{instr}\n\nRETRIEVED DATA:\n{data}\n\nNOTE:\n{note}\n")

    def specificity_completion_nudge(self, ctx, trajectory, judge):
        # If the PUBLIC instruction requires a dimension the note leaves VAGUE/absent, or a datum the agent
        # RETRIEVED but did not project into the note, re-enter with a VALUE-FREE completeness nudge (describe
        # the public slot, never the answer). The agent fills it from its OWN judgment + its OWN retrieved data.
        # Oracle-blind. Retention-protected (armed in before_final) so the additive rewrite cannot drop content.
        # Capped at 2 nudges.
        if not judge or not self.active or self.spec_nudges >= 2 or self._missing(ctx):
            return None
        gaps = self._detect_spec_gaps(ctx, trajectory, judge)
        if not gaps:
            return None
        return self._deliver_specificity(gaps, ctx)

    def _detect_spec_gaps(self, ctx, trajectory, judge):
        task = ctx.task
        note = self._read_deliverable(ctx)
        if not note or len(note.strip()) < 40:
            return None
        data = self._retrieved_summary(trajectory)
        try:
            from workspace_api import json_obj as _json_obj
            instr = str(task.get("goal") or "") + "\n" + str(task.get("context") or "")
            d = _json_obj(judge(self._SPEC_COMPLETION_PROMPT.format(instr=instr[:2500], data=data[:4000], note=note[:6000])))
            gaps = [g for g in ((d.get("gaps") if isinstance(d, dict) else None) or []) if isinstance(g, str) and g.strip()]
        except Exception as _je:
            # a JUDGE OUTAGE is NOT "no gaps" -- record it so this returning None is not read as "no opportunity".
            self._judge_errors.append("specificity_completion judge: %r" % (_je,))
            return None
        return gaps or None

    def _deliver_specificity(self, gaps, ctx):
        self.spec_nudges += 1
        fb = ("The public task requires the following, which your note does not yet concretely provide: %s. "
              "Complete EACH in your deliverable using your OWN clinical judgment and the data you have already "
              "retrieved -- KEEP everything you have already written and ADD the missing specifics on top. Then "
              "finish." % "; ".join(gaps[:8]))
        ctx.emit({"step": ctx.step, "event_type": "specificity_completion_nudge",
                  "gaps": gaps[:8], "attempt": self.spec_nudges, "status": "ok"})
        return {"feedback": fb}, fb

    def commitment_ledger_nudge(self, ctx, trajectory, judge):
        # Per-ITEM execution closure from the agent's OWN note (item-level rather than order-category level).
        # One judge call over the DELIVERABLE (the agent's own words) extracts the orders it presents as
        # place-now / already-done (NOT recommend / consider / if-conditional). Each such commitment must map to
        # a READ-BACK-CONFIRMED landed order (the same verified-receipt rule the freshness check uses). Any committed
        # order with no landed effect -> constrained re-entry: place the create tool call for it, or withdraw
        # it citing a specific patient finding -- NO soften-to-recommend escape. Authorization is the AGENT's
        # claim-vs-state, so it applies even when the PUBLIC task only
        # says determine/assess. Oracle-blind (agent note + agent's own landed orders; harness names no test).
        # Capped at 6 nudges.
        if not judge or not self.active or self.ledger_nudges >= 6 or self._missing(ctx):
            return None
        unmatched = self._detect_commitment_unmatched(ctx, trajectory, judge)
        if not unmatched:
            return None
        return self._deliver_commitment(unmatched, ctx)

    def _detect_commitment_unmatched(self, ctx, trajectory, judge):
        # Constrained execution repair. Persist (bounded) so an unfulfilled commitment is not
        # dropped just because the agent softens the wording; the ONLY clean exits are place-the-order or an
        # evidence-cited withdrawal -- no free-text soften escape.
        # FREEZE the agent's execute_now/asserted_done commitments, so a later note-rewrite that merely
        # rewords "order X" -> "consider X" cannot make the unfulfilled commitment vanish.
        # APPEND-ONLY UNION: re-parse the note EACH round and UNION any NEW commitment into the frozen set, so a
        # commitment the agent adds LATER (e.g. after a specificity_completion rewrite turns "consider an ECG"
        # into "order a 12-lead ECG now") is absorbed and then enforced -- while a reword "order X" -> "consider
        # X" still cannot REMOVE an already-frozen commitment (union never deletes). This connects the two
        # checks: public specificity gap -> agent writes a concrete action -> ledger execution closure.
        note = self._read_deliverable(ctx)
        if note and len(note.strip()) >= 40:
            try:
                from workspace_api import json_obj as _json_obj
                d = _json_obj(judge(self._COMMITMENT_PROMPT.format(note=note[:6000])))
                if isinstance(d, dict):
                    if self._frozen_commits is None:
                        self._frozen_commits = []
                    _have = {c.strip().lower() for c in self._frozen_commits}
                    for k in ("execute_now", "asserted_done"):
                        for c in (d.get(k) or []):
                            if isinstance(c, str) and c.strip() and c.strip().lower() not in _have:
                                _have.add(c.strip().lower()); self._frozen_commits.append(c.strip())
            except Exception as _je:
                # judge outage: a swallowed failure here would make "commitments unknown" look like
                # "no commitments" -> the fabricated-completion case (agent claims orders it never placed)
                # slips through silently. Record it so it surfaces as a runtime_error (never a silent None).
                self._judge_errors.append("commitment_ledger extract judge: %r" % (_je,))
        if not self._frozen_commits:
            return None
        # SEMANTIC 1:1 match (GENERAL -- no hardcoded synonym table / lexical patch): ONE judge call decides
        # which frozen commitments no placed order fulfills. The judge handles clinical synonyms/abbreviations
        # for ANY term/task (TSH=Thyrotropin, echo=echocardiography...) and 1:1 (each order fulfills <=1 commit).
        avail = list(self._confirmed_order_displays(trajectory))
        if not avail:
            unmatched = list(self._frozen_commits)   # nothing placed -> every commitment is still open
        else:
            try:
                from workspace_api import json_obj as _json_obj
                _d = _json_obj(judge(self._MATCH_PROMPT.format(
                    commits="\n".join("- %s" % c for c in self._frozen_commits),
                    orders="\n".join("- %s" % o for o in avail))))
                if not isinstance(_d, dict):
                    return None
                _um = {str(x).strip().lower() for x in (_d.get("unfulfilled") or []) if isinstance(x, str) and x.strip()}
                unmatched = [c for c in self._frozen_commits if c.strip().lower() in _um]
            except Exception as _je:
                # judge outage during matching: RECORD it (so this is not read as "all fulfilled / no gap"),
                # but do not fire on it -- firing on every commitment when the judge is flaky risks
                # over-nudging a correct answer. The failure is surfaced, the decision is not changed.
                self._judge_errors.append("commitment_ledger match judge: %r" % (_je,))
                return None
        return unmatched or None

    def _deliver_commitment(self, unmatched, ctx):
        self.ledger_nudges += 1
        fb = ("You committed in your OWN plan to these specific actions, but no matching order is recorded in "
              "the chart: %s. Complete them NOW: for EACH, your next tool call must be the appropriate create "
              "tool call for that exact action you already chose. Do NOT reword the note to 'recommend' / "
              "'consider' -- that leaves the action undone and does not resolve this. The only acceptable "
              "alternative to placing an order is to state, citing a SPECIFIC patient finding you retrieved, why "
              "it is contraindicated. Place the order(s) now." % "; ".join(unmatched[:8]))
        ctx.emit({"step": ctx.step, "event_type": "commitment_ledger_nudge",
                  "unmatched": unmatched[:8], "frozen": len(self._frozen_commits or []),
                  "attempt": self.ledger_nudges, "status": "ok"})
        return {"feedback": fb}, fb

    @staticmethod
    def _grounded_fact_tokens(text, trajectory):
        # Cheap grounded-fact vocabulary: the DISPLAY NAMES the agent actually RETRIEVED (subject-bound
        # reads) that ALSO appear in version A. These are patient facts A grounded; if B drops one, that is a
        # candidate regression to CONFIRM with the judge. Names come from the agent's OWN retrievals -> oracle-
        # blind (no gold/checkpoint). Used only to decide WHETHER to spend a judge call, never to block alone.
        summary = DeliverableMiddleware._retrieved_summary(trajectory)  # "…retrieved:\n<name>; <name>; …"
        if not summary:
            return []
        low = (text or "").lower()
        toks, seen = [], set()
        for nm in summary.split("\n", 1)[-1].split(";"):
            nm = nm.strip()
            k = nm.lower()
            if len(nm) >= 4 and k not in seen and k in low:
                seen.add(k); toks.append(nm)
            if len(toks) >= 60:
                break
        return toks

    _RETENTION_PROMPT = (
        "You compare COMPLETENESS between two versions of the SAME clinician note. Judge ONLY what is present "
        "or dropped, NEVER clinical correctness, and never invent an item that is not in the OLD note. Below "
        "are the OLD note and the NEW note the clinician just rewrote. List the concrete clinical TOPICS and "
        "the specific grounded PATIENT FACTS (named labs / vitals / results / orders / findings, with their "
        "values) that the OLD note addressed but the NEW note DROPPED -- i.e. the NEW note no longer states or "
        "addresses them. Ignore pure rewording that preserves the same topic/fact. Reply STRICT JSON "
        "{{\"dropped\": [\"<short label>\", ..]}} (empty list if the NEW note preserves everything the OLD "
        "note covered).\n\nOLD NOTE:\n{old}\n\nNEW NOTE:\n{new}\n")

    def retention_check(self, ctx, action, trajectory, judge):
        # Artifact retention: a correction nudge armed version A (arm_retention). The agent's follow-up
        # write B is a FULL 'w' overwrite that may ADD one section while DROPPING correct ones. Accept B only
        # if it does NOT regress -- i.e. it preserves every grounded patient fact + clinical topic A already
        # had. On regression, KEEP A on disk (caller does NOT execute this write) and ask for an ADDITIVE
        # localized patch. Oracle-blind: compares the agent's OWN two versions + the agent's OWN retrieved-data
        # names; authors no clinical content (echoes back only labels drawn from A). Bounded; fails OPEN.
        A = getattr(self, "_retain_A", None)
        if not self.active or not A or not A.strip() or not self.is_required_write(action):
            return None                                    # not armed / no prior version / not the deliverable
        if self.retention_blocks >= 2:                     # cap: never trap the agent in a rewrite loop
            self._retain_A = None
            return None
        B = str(((action.get("args") or {}).get("content")) or "")
        if not B.strip():
            return None                                    # empty write handled by pre_final / enforce
        # CHEAP structural pre-filter: only PAY for a judge call when B looks like it might have dropped
        # something. Suspicious = it lost a grounded-fact TOKEN A carried, OR it shrank substantially.
        dropped_tok = [f for f in self._grounded_fact_tokens(A, trajectory) if f.lower() not in B.lower()]
        shrank = len(B.strip()) < 0.6 * len(A.strip())
        if not dropped_tok and not shrank:
            self._retain_A = None                          # B keeps every grounded token + length -> accept
            return None
        # CONFIRM with ONE semantic judge (paraphrase-safe): does B actually DROP a topic/fact A had?
        try:
            from workspace_api import json_obj as _json_obj
            d = _json_obj(judge(self._RETENTION_PROMPT.format(old=A[:6000], new=B[:6000]))) if judge else {}
            dropped = [x for x in ((d.get("dropped") if isinstance(d, dict) else None) or [])
                       if isinstance(x, str) and x.strip()]
        except Exception:
            self._retain_A = None                          # judge error -> FAIL-OPEN (never trap), accept B
            return None
        if not dropped:
            self._retain_A = None                          # judge confirms no regression -> accept B
            return None
        # REGRESSION: keep A on disk (caller skips the write), request an ADDITIVE patch. _retain_A stays set
        # so the NEXT rewrite is compared against the SAME protected A (B never landed).
        self.retention_blocks += 1
        ctx.emit({"step": ctx.step, "event_type": "deliverable_retention_block",
                  "dropped": dropped[:8], "attempt": self.retention_blocks,
                  "b_shorter": shrank, "status": "ok"})
        fb = ("Your rewrite of the deliverable DROPPED content your previous version already contained: %s. "
              "Do NOT remove what you had already written. Re-issue write_file with EXACTLY path=\"%s\": keep "
              "everything from your previous version (all of the above must still be present) and ADD the new "
              "material on top of it, then finish." % ("; ".join(dropped[:8]), self.path))
        return {"feedback": fb}, "retention_block"

    def pre_final_nudge(self, ctx):
        # before accepting a final answer, force the deliverable to be written first (up to 3x);
        # returns (last_res, last_obs) to inject + signal continue, or None to allow finishing
        if not self.active or self.nudges >= 3 or not self._missing(ctx):
            return None
        self.nudges += 1
        fb = ("STOP. The task REQUIRES a deliverable file and it is NOT written yet. You MUST "
              "call write_file NOW with EXACTLY path=\"%s\" (this EXACT filename, no other name) "
              "and content = your full clinical assessment and management plan. Do not answer in "
              "chat. Do not use any other filename. After write_file succeeds, then finish." % self.path)
        ctx.emit({"step": ctx.step, "event_type": "deliverable_nudge", "path": self.path,
                  "attempt": self.nudges, "status": "ok"})
        return {"feedback": fb}, fb

    # ------------------------------------------------------------------ middleware hooks (runner/middleware_host.py)
    def before_model(self, ctx):
        r = self.budget_warning(ctx)
        return None if r is None else Observation(r[0], r[1])

    def before_final(self, ctx, answer):
        # final-gate ORDER: the deliverable exists -> it is fresh -> it is specific -> its commitments landed.
        # One fires per turn; retention is armed on each correction nudge so the rewrite it triggers is checked.
        r = self.pre_final_nudge(ctx)
        if r is not None:
            return Revision("deliverable", (r[0] or {}).get("feedback", ""), r[1])
        judge = ctx.judge
        trajectory = list(ctx.trajectory)   # one read of the trajectory for this hook (a nudge that fires returns)
        for mechanism, stage, nudge in (("deliverable_freshness", "freshness", self.deliverable_freshness_nudge),
                                        ("specificity_completion", "specificity_completion", self.specificity_completion_nudge),
                                        ("commitment_ledger", "commitment_ledger", self.commitment_ledger_nudge)):
            try:
                r = nudge(ctx, trajectory, judge)
            except Exception as e:
                r = None
                ctx.error("%s_nudge: %r" % (mechanism, e))
            if r is None:
                continue
            self.arm_retention(ctx)
            return Revision(stage, (r[0] or {}).get("feedback", ""), r[1])
        return None

    def before_tool(self, ctx, action):
        if not (action.get("tool") == "write_file" and (action.get("args") or {}).get("content")):
            return None
        try:
            r = self.retention_check(ctx, action, list(ctx.trajectory), ctx.judge)
        except Exception as e:
            r = None
            ctx.error("retention_check: %r" % (e,))
        if r is None:
            return None
        return Revision("retention", (r[0] or {}).get("feedback", ""), r[1])

    def exempt_from_budget(self, ctx, action):
        # the one exact deliverable write may still run once the tool budget is spent
        return self.is_required_write(action) and self._missing(ctx)

    def after_episode(self, ctx):
        # guarantee ONE write attempt if the required file is still missing, then report it honestly if the
        # agent still did not write it (the harness never fabricates or renames a deliverable)
        if not self.active:
            return
        ws = ctx.output_files() is not None
        if ws and self._missing(ctx):
            fb = ("You must NOW save the required deliverable. Call write_file with EXACTLY path=\"%s\" and "
                  "content = your full clinical assessment and management plan." % self.path)
            ctx.forced_deliverable_write(fb)
        out = ctx.output_files()
        if ws and out is not None and out.get("exists") and self._missing(ctx):
            ctx.emit({"step": ctx.max_steps, "event_type": "deliverable_missing",
                      "path": posixpath.basename(self.path or ""), "status": "ok"})

    def deferred_errors(self):
        return list(self._judge_errors)
