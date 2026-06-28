#!/bin/bash
# Formal driver — the THREE datasets run in PARALLEL (different backends: PB->FHIR, MedCTA->VLM/gateway,
# HAB->portal; disjoint output dirs), each stream INTERNALLY SEQUENTIAL over its modes so off/enforce never
# cross-contaminate the same backend. After every batch the canonical post-hoc Governance is rescored and
# VERIFIED (count==limit else the stream FAILs). When all three finish: cmp_report res6 --formal.
# Agent=gpt-5.5; harness judge=gpt-5.4-mini (independent); native GAcc + Governance rescore judge=gpt-5.4.
set -uo pipefail
cd /hpc2hdd/home/ce483/Medical_harness
PY=~/.conda/envs/medicalharness/bin/python
export MH_OPENAI_BASE=https://www.micuapi.ai MH_API_MODEL=gpt-5.5
FHIR=http://127.0.0.1:38080/fhir; PORTAL=http://127.0.0.1:3002
RESCORE_JUDGE=gpt-5.4; LIMIT=10
# declared task-universe manifest (the first LIMIT task_ids run_batch selects per bench) -> cmp_report
# requires paired_ids == declared, so a task that fails in ALL modes cannot silently shrink n.
$PY - "$LIMIT" "$(git rev-parse --short HEAD)" << 'PYMAN'
import json,sys
LIM=int(sys.argv[1]); SHA=sys.argv[2]
modes={"PhysicianBench":["off","enforce"],"MedCTA":["off","observe","assist","enforce"],"HealthAdminBench":["off","enforce"]}
decl={b:[json.loads(l)["task_id"] for l in open("benchmark_dataprocess/%s/tasks_unified.jsonl"%b)][:LIM] for b in modes}
json.dump({"git_sha":SHA,"limit":LIM,"declared":decl,"modes":modes}, open("res6_manifest.json","w"), indent=1)
print("[manifest] res6_manifest.json sha=%s limit=%d"%(SHA,LIM))
PYMAN
echo "[start $(date) host=$(hostname) sha=$(git rev-parse --short HEAD)] PARALLEL by dataset"
curl -s -m8 -o /dev/null -w "FHIR %{http_code}\n" $FHIR/metadata || true
curl -s -m8 -o /dev/null -w "PORTAL %{http_code}\n" $PORTAL/emr/denied || true

rescore () { $PY runner/rescore_judges.py "$1/gpt5" --bench "$2" --judge-model $RESCORE_JUDGE >/dev/null 2>&1; }
verify_rescored () {
  local n; n=$(ls "$1"/gpt5/*/result.rescored.json 2>/dev/null | wc -l)
  if [ "$n" -ne "$2" ]; then echo "[FATAL] $1: rescored $n != $2"; return 1; fi
  echo "[verify $1: $n/$2 rescored]"
}

pb_stream () { set -e
  for M in off enforce; do
    MH_HARNESS_MODE=$M $PY runner/run_batch.py --bench PhysicianBench --agent gpt5 --limit $LIMIT \
       --fhir-base $FHIR --reset-mode restore_pristine --out res6_pb_$M/ > res6_pb_$M.log 2>&1
    rescore res6_pb_$M PhysicianBench; verify_rescored res6_pb_$M $LIMIT; echo "[done PB $M $(date)]"
  done
}
mcta_stream () { set -e
  for M in off observe assist enforce; do
    local HJ=""; if [ "$M" != "off" ]; then HJ="gpt-5.4-mini"; fi
    MH_HARNESS_MODE=$M MH_HARNESS_JUDGE_MODEL=$HJ MH_TOOL_MODE=real MH_GACC=1 MH_GACC_MODEL=gpt-5.4 \
       $PY runner/run_batch.py --bench MedCTA --agent gpt5 --limit $LIMIT --out res6_mcta_$M/ > res6_mcta_$M.log 2>&1
    rescore res6_mcta_$M MedCTA; verify_rescored res6_mcta_$M $LIMIT; echo "[done MCTA $M $(date)]"
  done
}
hab_stream () { set -e
  for M in off enforce; do
    local HJ=""; if [ "$M" = "enforce" ]; then HJ="gpt-5.4-mini"; fi
    MH_HARNESS_MODE=$M MH_HARNESS_JUDGE_MODEL=$HJ MH_GUI_MODE=real MH_PORTAL_BASE=$PORTAL MH_GACC=1 MH_GACC_MODEL=gpt-5.4 \
       $PY runner/run_batch.py --bench HealthAdminBench --agent gpt5 --limit $LIMIT --out res6_hab_$M/ > res6_hab_$M.log 2>&1
    rescore res6_hab_$M HealthAdminBench; verify_rescored res6_hab_$M $LIMIT; echo "[done HAB $M $(date)]"
  done
}

pb_stream &   PB=$!
mcta_stream & MC=$!
hab_stream &  HB=$!
RC=0
wait $PB; r=$?; [ $r -ne 0 ] && { RC=1; echo "[stream PB FAILED rc=$r]"; }
wait $MC; r=$?; [ $r -ne 0 ] && { RC=1; echo "[stream MCTA FAILED rc=$r]"; }
wait $HB; r=$?; [ $r -ne 0 ] && { RC=1; echo "[stream HAB FAILED rc=$r]"; }
echo "[ALL STREAMS DONE $(date) rc=$RC]"
if [ "$RC" -ne 0 ]; then echo "[FATAL] one or more dataset streams failed -> no formal report"; exit "$RC"; fi
$PY cmp_report.py res6 --formal
