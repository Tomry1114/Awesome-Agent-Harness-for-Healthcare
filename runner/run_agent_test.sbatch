#!/bin/bash
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH -J qwen_agent
#SBATCH -o /hpc2hdd/home/ce483/Medical_harness/runner/agent_test.log
cd /hpc2hdd/home/ce483/Medical_harness
export MH_TOOL_MODE=real
echo "[start] $(date) node=$(hostname)"
~/.conda/envs/medicalharness/bin/python runner/agent_test_driver.py MedCTA MCTA-0 2>&1 | grep -vE "torch_dtype|deprecated|generation flags|TRANSFORMERS_VERBOSITY"
echo "[done] rc=$?"
