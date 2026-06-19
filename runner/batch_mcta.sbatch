#!/bin/bash
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH -J b_mcta
#SBATCH -o /hpc2hdd/home/ce483/Medical_harness/runner/batch_mcta.log
cd /hpc2hdd/home/ce483/Medical_harness
echo "[start] $(date) node=$(hostname)"
export MH_TOOL_MODE=real
~/.conda/envs/medicalharness/bin/python runner/run_batch.py --bench MedCTA --agent qwen --limit 10 --out results_v0/MedCTA/ 2>&1 | grep -vE "torch_dtype|deprecated|generation flags|TRANSFORMERS_VERBOSITY|Setting .pad_token|Some weights"
echo "[done] rc=$?"
