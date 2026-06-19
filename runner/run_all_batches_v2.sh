#!/bin/bash
cd /hpc2hdd/home/ce483/Medical_harness
PY=~/.conda/envs/medicalharness/bin/python
echo "[start $(date) node=$(hostname)]"
echo "=========== PB (FHIR fixed, no judge) $(date) ==========="
curl -s -m8 -o /dev/null -w "FHIR %{http_code}\n" http://10.120.31.247:38080/fhir/metadata
$PY runner/run_batch.py --bench PhysicianBench --agent qwen --limit 10 --fhir-base http://10.120.31.247:38080/fhir --reset-mode none --out results_v0b/PhysicianBench/
echo "=========== MedCTA (MH_JUDGE=qwen) $(date) ==========="
MH_TOOL_MODE=real MH_JUDGE=qwen $PY runner/run_batch.py --bench MedCTA --agent qwen --limit 10 --out results_v0b/MedCTA/
echo "=========== HAB (MH_JUDGE=qwen, real portal) $(date) ==========="
curl -s -m8 -o /dev/null -w "PORTAL %{http_code}\n" http://10.120.31.247:3002/emr/denied
MH_GUI_MODE=real MH_PORTAL_BASE=http://10.120.31.247:3002 MH_JUDGE=qwen $PY runner/run_batch.py --bench HealthAdminBench --agent qwen --limit 6 --out results_v0b/HealthAdminBench/
echo "[ALL DONE $(date)]"
