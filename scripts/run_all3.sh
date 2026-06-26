#!/usr/bin/env bash
cd ~/Medical_harness
PY=~/.conda/envs/medicalharness/bin/python
COMMON="MH_OPENAI_MODEL=gpt-5.5 MH_JUDGE_MODEL=gpt-5.4 MH_RUBRIC_JUDGE=1 MH_GACC=1 MH_GACC_MODEL=gpt-5.4 MH_VERIFICATION_JUDGE=1 MH_GOV_JUDGE=1"
echo "[$(date +%T)] MedCTA 10"
rm -rf results_mcta_final
env $COMMON MH_MM_JUDGE=1 MH_MM_JUDGE_MODEL=gpt-5.4 \
  $PY runner/run_batch.py --bench MedCTA --agent gpt5 --limit 10 --max-steps 12 --out results_mcta_final > results_mcta_final.log 2>&1
echo "[$(date +%T)] MedCTA done: $(ls results_mcta_final/gpt5/MCTA-*/result.json 2>/dev/null|wc -l)"
echo "[$(date +%T)] PhysicianBench 10 (FHIR)"
rm -rf results_pb_final
env $COMMON \
  $PY runner/run_batch.py --bench PhysicianBench --agent gpt5 --limit 10 --max-steps 14 --fhir-base http://localhost:38080/fhir --reset-mode none --out results_pb_final > results_pb_final.log 2>&1
echo "[$(date +%T)] PB done: $(ls results_pb_final/gpt5/PB-*/result.json 2>/dev/null|wc -l)"
echo "[$(date +%T)] HealthAdminBench 10"
rm -rf results_hab_final
env $COMMON \
  $PY runner/run_batch.py --bench HealthAdminBench --agent gpt5 --limit 10 --max-steps 12 --out results_hab_final > results_hab_final.log 2>&1
echo "[$(date +%T)] HAB done: $(ls results_hab_final/gpt5/HAB-*/result.json 2>/dev/null|wc -l)"
for d in mcta:MedCTA pb:PhysicianBench hab:HealthAdminBench; do
  MH_JUDGE_MODEL=gpt-5.4 $PY runner/aggregate_report.py results_${d%%:*}_final/gpt5 --bench ${d##*:} > /dev/null 2>&1
done
echo "[$(date +%T)] ALL DONE + aggregated"
