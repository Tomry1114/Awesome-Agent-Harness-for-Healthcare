#!/bin/bash
# Formal driver — matches the cmp_report contract: MedCTA runs ALL FOUR modes (off/observe/assist/enforce);
# PB/HAB run off/enforce. After each batch the canonical post-hoc Governance is rescored (writes
# result.rescored.json per task) so cmp_report reads the SAME Governance the paper reports. Every batch is
# VERIFIED: the number of result.rescored.json files MUST equal the batch limit, else the run FAILS loudly.
# Fail on unset vars / pipe errors / any command error. Agent=gpt-5.5; harness judge=gpt-5.4-mini
# (independent); native GAcc + rescore judge=gpt-5.4.
set -euo pipefail
cd /hpc2hdd/home/ce483/Medical_harness
PY=~/.conda/envs/medicalharness/bin/python
export MH_OPENAI_BASE=https://www.micuapi.ai MH_API_MODEL=gpt-5.5
FHIR=http://127.0.0.1:38080/fhir; PORTAL=http://127.0.0.1:3002
RESCORE_JUDGE=gpt-5.4
LIMIT=10
echo "[start $(date) host=$(hostname) sha=$(git rev-parse --short HEAD)]"
curl -s -m8 -o /dev/null -w "FHIR %{http_code}\n" $FHIR/metadata || true
curl -s -m8 -o /dev/null -w "PORTAL %{http_code}\n" $PORTAL/emr/denied || true

rescore () { $PY runner/rescore_judges.py "$1/gpt5" --bench "$2" --judge-model $RESCORE_JUDGE >/dev/null 2>&1; }

# After rescore, EVERY task must carry a result.rescored.json. count == limit, else FAIL.
verify_rescored () {
  local dir="$1" exp="$2" n
  n=$(ls "$dir"/gpt5/*/result.rescored.json 2>/dev/null | wc -l)
  if [ "$n" -ne "$exp" ]; then
    echo "[FATAL] $dir: rescored $n != expected $exp result.rescored.json files"; exit 1
  fi
  echo "[verify $dir: $n/$exp rescored]"
}

# ---- PhysicianBench (structured_record; no semantic judge) ----
for M in off enforce; do
  MH_HARNESS_MODE=$M $PY runner/run_batch.py --bench PhysicianBench --agent gpt5 --limit $LIMIT \
     --fhir-base $FHIR --reset-mode none --out res6_pb_$M/ >/dev/null 2>&1
  rescore res6_pb_$M PhysicianBench
  verify_rescored res6_pb_$M $LIMIT
  echo "[done PB $M $(date)]"
done

# ---- MedCTA (perceptual) — ALL FOUR MODES (matches cmp_report) ----
for M in off observe assist enforce; do
  HJ=""
  if [ "$M" != "off" ]; then HJ="gpt-5.4-mini"; fi
  MH_HARNESS_MODE=$M MH_HARNESS_JUDGE_MODEL=$HJ MH_TOOL_MODE=real MH_GACC=1 MH_GACC_MODEL=gpt-5.4 \
     $PY runner/run_batch.py --bench MedCTA --agent gpt5 --limit $LIMIT --out res6_mcta_$M/ >/dev/null 2>&1
  rescore res6_mcta_$M MedCTA
  verify_rescored res6_mcta_$M $LIMIT
  echo "[done MCTA $M $(date)]"
done

# ---- HealthAdminBench (interactive_gui) ----
for M in off enforce; do
  HJ=""
  if [ "$M" = "enforce" ]; then HJ="gpt-5.4-mini"; fi
  MH_HARNESS_MODE=$M MH_HARNESS_JUDGE_MODEL=$HJ MH_GUI_MODE=real MH_PORTAL_BASE=$PORTAL MH_GACC=1 MH_GACC_MODEL=gpt-5.4 \
     $PY runner/run_batch.py --bench HealthAdminBench --agent gpt5 --limit $LIMIT --out res6_hab_$M/ >/dev/null 2>&1
  rescore res6_hab_$M HealthAdminBench
  verify_rescored res6_hab_$M $LIMIT
  echo "[done HAB $M $(date)]"
done

echo "[ALL DONE $(date)] -> cmp_report.py res6 --formal"
/hpc2hdd/home/ce483/.conda/envs/medicalharness/bin/python cmp_report.py res6 --formal
