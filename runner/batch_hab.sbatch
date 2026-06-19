#!/bin/bash
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH -J b_hab
#SBATCH -o /hpc2hdd/home/ce483/Medical_harness/runner/batch_hab.log
cd /hpc2hdd/home/ce483/Medical_harness
echo "[start] $(date) node=$(hostname)"
export MH_GUI_MODE=real
export MH_PORTAL_BASE=http://10.120.31.247:3002
curl -s -m 8 -o /dev/null -w "PORTAL reach: %{http_code}\n" $MH_PORTAL_BASE/emr/denied
~/.conda/envs/medicalharness/bin/python runner/run_batch.py --bench HealthAdminBench --agent qwen --limit 10 --out results_v0/HealthAdminBench/ 2>&1 | grep -vE "torch_dtype|deprecated|generation flags|TRANSFORMERS_VERBOSITY|Setting .pad_token|Some weights"
echo "[done] rc=$?"
