#!/bin/bash
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH -J pb_agent
#SBATCH -o /hpc2hdd/home/ce483/Medical_harness/runner/pb_agent.log
cd /hpc2hdd/home/ce483/Medical_harness
export FHIR_BASE_URL=http://10.120.31.247:38080/fhir
echo "[start] $(date) node=$(hostname)"
curl -s -m 8 -o /dev/null -w "FHIR reach: %{http_code}\n" $FHIR_BASE_URL/metadata
~/.conda/envs/medicalharness/bin/python runner/agent_test_driver.py PhysicianBench PB-aberrant_drug_screen 2>&1 | grep -vE "torch_dtype|deprecated|generation flags|TRANSFORMERS_VERBOSITY"
echo "[done] rc=$?"
