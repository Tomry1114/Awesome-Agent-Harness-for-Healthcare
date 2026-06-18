#!/bin/bash
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH -J hab_agent
#SBATCH -o /hpc2hdd/home/ce483/Medical_harness/runner/hab_agent.log
cd /hpc2hdd/home/ce483/Medical_harness
export MH_PORTAL_BASE=http://10.120.31.247:3002
export MH_GUI_MODE=real   # GPU node has launchable chromium; use the real portal
TASK=${1:-HAB-denial-easy-1}
echo "[start] $(date) node=$(hostname) task=$TASK"
curl -s -m 8 -o /dev/null -w "PORTAL reach: %{http_code}\n" $MH_PORTAL_BASE/emr/denied
python -c "import playwright; print('playwright', playwright.__version__)" 2>/dev/null
~/.conda/envs/medicalharness/bin/python runner/agent_test_driver.py HealthAdminBench $TASK 14 2>&1 | grep -vE "torch_dtype|deprecated|generation flags|TRANSFORMERS_VERBOSITY"
echo "[done] rc=$?"
