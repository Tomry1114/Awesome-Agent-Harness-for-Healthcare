#!/bin/bash
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH -J mcta_prof
#SBATCH -o /hpc2hdd/home/ce483/Medical_harness/runner/medcta_profile.log
cd /hpc2hdd/home/ce483/Medical_harness
echo "[start] $(date) node=$(hostname)"
~/.conda/envs/medicalharness/bin/python runner/medcta_profile.py 2>&1 | grep -vE "torch_dtype|deprecated|generation flags|TRANSFORMERS_VERBOSITY|Setting .pad_token"
echo "[done] rc=$?"
