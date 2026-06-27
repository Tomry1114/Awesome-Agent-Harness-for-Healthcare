#!/bin/bash
# Harness vs no-harness comparison: 3 datasets x {off, enforce} x 10 tasks. Fair: same agent (gpt-5.5),
# same task selection, same backends; only MH_HARNESS_MODE differs. Harness judge is INDEPENDENT (gpt-5.4-mini != agent).
set -u
cd /hpc2hdd/home/ce483/Medical_harness
PY=~/.conda/envs/medicalharness/bin/python
export MH_OPENAI_BASE=https://www.micuapi.ai
export MH_API_MODEL=gpt-5.5
FHIR=http://127.0.0.1:38080/fhir
PORTAL=http://127.0.0.1:3002
echo "[start $(date) host=$(hostname)]"
curl -s -m8 -o /dev/null -w "FHIR %{http_code}\n" $FHIR/metadata
curl -s -m8 -o /dev/null -w "PORTAL %{http_code}\n" $PORTAL/emr/denied

run_one () {  # $1=bench $2=mode $3=outdir ; extra env passed inline by caller
  local bench=$1 mode=$2 out=$3
  echo "=================== $bench  mode=$mode  -> $out  $(date) ==================="
  MH_HARNESS_MODE=$mode "${@:4}" $PY runner/run_batch.py --bench $bench --agent gpt5 --limit 10 \
     --out $out/ 2>&1
}

# ---- PhysicianBench (FHIR, structured_record; no semantic judge needed) ----
for M in off enforce; do
  MH_HARNESS_MODE=$M $PY runner/run_batch.py --bench PhysicianBench --agent gpt5 --limit 10 \
     --fhir-base $FHIR --reset-mode none --out res5_pb_$M/ 2>&1
  echo "[done PB $M $(date)]"
done

# ---- MedCTA (perceptual; native judge gpt-5.4; harness judge independent) ----
for M in off enforce; do
  HJ=""; [ "$M" = "enforce" ] && HJ="gpt-5.4-mini"
  MH_HARNESS_MODE=$M MH_HARNESS_JUDGE_MODEL=$HJ MH_TOOL_MODE=real MH_JUDGE=gpt-5.4 \
     $PY runner/run_batch.py --bench MedCTA --agent gpt5 --limit 10 --out res5_mcta_$M/ 2>&1
  echo "[done MCTA $M $(date)]"
done

# ---- HealthAdminBench (interactive_gui; native judge gpt-5.4; harness judge independent) ----
for M in off enforce; do
  HJ=""; [ "$M" = "enforce" ] && HJ="gpt-5.4-mini"
  MH_HARNESS_MODE=$M MH_HARNESS_JUDGE_MODEL=$HJ MH_GUI_MODE=real MH_PORTAL_BASE=$PORTAL MH_JUDGE=gpt-5.4 \
     $PY runner/run_batch.py --bench HealthAdminBench --agent gpt5 --limit 10 --out res5_hab_$M/ 2>&1
  echo "[done HAB $M $(date)]"
done
echo "[ALL DONE $(date)]"
