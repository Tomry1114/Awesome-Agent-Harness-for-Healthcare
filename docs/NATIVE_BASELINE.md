# Native agent and Base Harness comparison

`--execution-mode native` runs the benchmark's workspace Agent with its normal prompt and tool protocol, but does not construct the Harness kernel, load host middleware, apply obligation/collection nudges, or grant recovery actions. The same environment adapter executes the Agent's actions. The trajectory, benchmark-native outcome checks, and offline process scorers still run. `--execution-mode harness` (the default) runs H0 with the kernel and middleware. This is an execution intervention, not a change to GT or scoring rules.

Use the same task IDs, split, agent model, prompt/workspace, tool backend, judge configuration, environment assets, `--max-steps`, and reset policy in both arms. Use separate output roots; the batch runner refuses to overwrite a stored result from the opposite mode. Its resume key includes `execution_mode` for native runs. Keep environment health and judge failures separate from Agent failures. For HealthAdminBench, the H0 arm may use an additional recovery-action reserve; inspect each result's `budget` before claiming equal total action cost.

For example, after configuring the environment and judge models, and preparing one task ID per line in `selected_ids.txt`:

```bash
python3 runner/run_batch.py --bench MedCTA --agent gpt5 --task-ids-file selected_ids.txt \
  --max-steps 12 --execution-mode native --out results/native/
python3 runner/run_batch.py --bench MedCTA --agent gpt5 --task-ids-file selected_ids.txt \
  --max-steps 12 --execution-mode harness --out results/h0/
```

Use the same pattern for PhysicianBench and HealthAdminBench, with their required FHIR/portal services and reset settings. The public repository does not include the upstream benchmark runtimes or large assets; install them as described in `benchmark/README.md` before a live run.

For each task, compare `result.json` (`run_config.execution_mode`, `harness`, `budget`, native Outcome checkpoints, `outcome_status`, `qualification`) and `trajectory.jsonl` (tool calls, observations, final answer). The seven process scores are reported separately from native Outcome. A successful interface smoke test is not evidence of benchmark-level improvement.
