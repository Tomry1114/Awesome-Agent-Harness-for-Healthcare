#!/bin/bash
cd /hpc2hdd/home/ce483/Medical_harness
PY=~/.conda/envs/medicalharness/bin/python
echo "[start $(date) node=$(hostname)]"
echo "=========== MCTA $(date) ==========="
MH_TOOL_MODE=real $PY runner/run_batch.py --bench MedCTA --agent qwen --limit 10 --out results_v0/MedCTA/
echo "=========== PB $(date) ==========="
curl -s -m8 -o /dev/null -w "FHIR reach %{http_code}\n" http://10.120.31.247:38080/fhir/metadata
$PY runner/run_batch.py --bench PhysicianBench --agent qwen --limit 10 --fhir-base http://10.120.31.247:38080/fhir --reset-mode none --out results_v0/PhysicianBench/
echo "=========== HAB $(date) ==========="
curl -s -m8 -o /dev/null -w "PORTAL reach %{http_code}\n" http://10.120.31.247:3002/emr/denied
MH_GUI_MODE=real MH_PORTAL_BASE=http://10.120.31.247:3002 $PY runner/run_batch.py --bench HealthAdminBench --agent qwen --limit 10 --out results_v0/HealthAdminBench/
echo "[ALL DONE $(date)]"
