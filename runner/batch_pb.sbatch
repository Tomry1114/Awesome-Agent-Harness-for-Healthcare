#!/bin/bash
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH -J b_pb
#SBATCH -o /hpc2hdd/home/ce483/Medical_harness/runner/batch_pb.log
cd /hpc2hdd/home/ce483/Medical_harness
echo "[start] $(date) node=$(hostname)"
export FHIR=http://10.120.31.247:38080/fhir
curl -s -m 8 -o /dev/null -w "FHIR reach: %{http_code}\n" $FHIR/metadata
~/.conda/envs/medicalharness/bin/python runner/run_batch.py --bench PhysicianBench --agent qwen --limit 10 --fhir-base $FHIR --reset-mode none --out results_v0/PhysicianBench/ 2>&1 | grep -vE "torch_dtype|deprecated|generation flags|TRANSFORMERS_VERBOSITY|Setting .pad_token|Some weights"
echo "[done] rc=$?"
