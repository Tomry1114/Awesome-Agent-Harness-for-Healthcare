# User-study benchmarks

This directory contains the conversion and runtime adapters for three additional evaluation sources. Upstream records, images, browser services, generated tasks, model weights and review data are installed locally; they are not part of this repository. The original 342-task train/validation/test split does not include these sources.

| Evaluation source | Prepared task universe | Outcome | Runtime |
|---|---:|---|---|
| MedRBench | 3,367 variants from 1,453 cases | Reference-grounded Judge; expert review can be added | Case text plus optional one-turn/free-turn patient information service |
| MeissaEval | 6,676 screened PathVQA test questions in the local conversion | Reference-grounded Judge | Original image, zoom and optional live segmentation workers |
| MedCUABench | 432 intent/step variants from 216 base tasks | Official completion and reward checker | Pinned BrowserGym/Playwright source, screenshot-only agent observations |

The counts describe the prepared local conversions, not files shipped in this checkout or completed agent runs. MeissaEval removes detected image/near-image overlap candidates; it does not establish a provenance-verified held-out split. Meissa-SFT contains training trajectories and is **not** counted as an evaluation benchmark. A tool response or a generated mask alone does not establish clinical correctness.

## Prepare sources

Run commands from the repository root. Use the revisions recorded in [benchmark/README.md](../benchmark/README.md).

```bash
git clone https://github.com/MAGIC-AI4Med/MedRBench.git user_study/MedR-Bench
git -C user_study/MedR-Bench checkout ff60ab440afd2f2bc0c603b3a65d715ec83138a7
python3 user_study/scripts/prepare.py --dataset medr --medr-protocol all

git clone https://github.com/Jia7878/MedCUA-BenchMark.git user_study/MedCUA-BenchMark
git -C user_study/MedCUA-BenchMark checkout 159c7681cbc84d446ceafa8046bd498036809b70
# Install the source project's browser dependencies in an isolated Python environment.
export MH_MEDCUA_PYTHON="$PWD/user_study/runtime/medcua_env/bin/python"
python3 user_study/scripts/prepare.py --dataset medcua
```

The MedCUA conversion verifies the tracked upstream files against `config/medcua_source.json`; editing the source or using a different revision fails closed. HTML tasks require a working browser. OpenEMR and OHIF tasks need their separate upstream services. A prepared task is not automatically runnable on every machine.

For MeissaEval, download the PathVQA parquet files at the revision in `Meissa-Evaluation/source/download_manifest.json` into `user_study/Meissa-Evaluation/source/data/`. Install Meissa-SFT's original `data/train.jsonl` and image assets under `user_study/Meissa-SFT/` for the overlap audit. Then run:

```bash
python3 user_study/scripts/prepare_pathvqa.py
python3 user_study/scripts/prepare.py --dataset meissa-eval \
  --meissa-eval-manifest user_study/Meissa-Evaluation/pathvqa_test.screened.jsonl
```

The PathVQA preparation checks each parquet SHA-256 against the manifest. Keep the generated audit and conversion reports with any published result. Live zoom uses the local image; SAM2 and BioMedParse require separately configured workers (`MH_MEISSA_SAM2_URL` and `MH_MEISSA_BIOMEDPARSE_URL`). Check `runner/adapters/user_study/meissa_image.py` for their request contract. Without those workers, a run cannot claim to have tested segmentation.

## Run and score

Set the same agent and Judge configuration described in the root README. MedR interactive tasks also need a patient-service model (`MH_MEDR_PATIENT_MODEL`, optional separate `MH_MEDR_PATIENT_BASE`). Keep all credentials in environment variables or ignored local files.

```bash
python3 runner/run_batch.py --bench MedRBench --study-protocol free_turn \
  --agent gpt5 --limit 1 --out results/medr/
python3 runner/run_batch.py --bench MeissaEval --agent gpt5 --limit 1 --out results/meissa/
python3 runner/run_batch.py --bench MedCUABench --agent gpt5 --limit 1 \
  --max-steps 30 --out results/medcua/
```

Each benchmark can run with `--execution-mode native` or `--execution-mode harness` for a paired native-agent/Base-Harness comparison. Compare the same prepared task IDs, model, services and Judge configuration. The Outcome remains separate from the seven process dimensions. `user_study/scripts/seven_dimensions.py` freezes and evaluates the optional seven-dimension admission profile; a missing or invalid lock is reported as incomplete rather than silently filling scores. Generate the lock only after code and scoring rules are finalized.

The test suite checks adapter contracts with local fixtures. It does not substitute for a real model run, GPU image-worker inference, upstream browser services, expert review or a clinical-validity study.
