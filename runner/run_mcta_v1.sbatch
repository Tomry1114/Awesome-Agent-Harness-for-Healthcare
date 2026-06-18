#!/bin/bash
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH -J mcta_v1
#SBATCH -o /hpc2hdd/home/ce483/Medical_harness/runner/mcta0_v1.log
cd /hpc2hdd/home/ce483/Medical_harness
export MH_TOOL_MODE=real
echo "[start] $(date) node=$(hostname)"
~/.conda/envs/AgentOCR/bin/python runner/run.py --bench MedCTA --task MCTA-0 --agent replay --out /tmp/mcta0_v1.json
echo "[done] rc=$?"
