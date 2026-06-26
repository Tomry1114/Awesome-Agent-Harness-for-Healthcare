cd ~/Medical_harness
OLD=$(cat ~/.xbai_key)
PY=/hpc2hdd/home/ce483/.conda/envs/medicalharness/bin/python
N=10
COMMON="OPENAI_API_KEY=$OLD MH_OPENAI_BASE=https://www.micuapi.ai MH_API_MODEL=gpt-5.5 \
  MH_JUDGE_KEY=$OLD MH_JUDGE_MODEL=gpt-5.4 MH_GACC=1 MH_GACC_MODEL=gpt-5.4 MH_MM_JUDGE=1 MH_MM_JUDGE_MODEL=gpt-5.4 \
  MH_VERIFICATION_JUDGE=1 MH_GOV_JUDGE=1 MH_VLM_API_KEY=$OLD MH_VLM_API_MODEL=gpt-5.5 \
  MH_HARNESS_JUDGE_MODEL=gpt-5.4 MH_GATEWAY_TIMEOUT=120 MH_GATEWAY_RETRIES=3"
rm -f res4_run.log
for MODE in off observe assist enforce; do
  OUT=res4_mcta_$MODE
  rm -rf $OUT
  echo "[run] mode=$MODE" >> res4_run.log
  env $COMMON MH_HARNESS_MODE=$MODE $PY runner/run_batch.py --bench MedCTA --max-steps 12 \
      --agent gpt5 --limit $N --out $OUT >> res4_run.log 2>&1
  env OPENAI_API_KEY=$OLD MH_OPENAI_BASE=https://www.micuapi.ai LLM_JUDGE_BACKEND=openai \
      OPENAI_BASE_URL=https://www.micuapi.ai/v1 LLM_JUDGE_MODEL=gpt-5.4 \
      $PY runner/rescore_judges.py $OUT/gpt5 --judge-model gpt-5.4 >> res4_run.log 2>&1
  $PY runner/aggregate_report.py $OUT/gpt5 > $OUT/report.stdout 2>> res4_run.log
  echo "[done] mode=$MODE" >> res4_run.log
done
echo "P4_DONE" >> res4_run.log
