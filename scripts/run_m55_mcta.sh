cd ~/Medical_harness
env MH_OPENAI_MODEL=gpt-5.5 MH_JUDGE_MODEL=gpt-5.4 MH_RUBRIC_JUDGE=1 MH_GACC=1 MH_GACC_MODEL=gpt-5.4 MH_VERIFICATION_JUDGE=1 MH_GOV_JUDGE=1 MH_MM_JUDGE=1 MH_MM_JUDGE_MODEL=gpt-5.4 MH_GATEWAY_TIMEOUT=120 MH_OPENAI_TIMEOUT=120 MH_GATEWAY_RETRIES=3 /hpc2hdd/home/ce483/.conda/envs/medicalharness/bin/python runner/run_batch.py --bench MedCTA --max-steps 12 --agent gpt5 --limit 5 --out res_m55_mcta > res_m55_mcta.log 2>&1
echo DONE_m55_mcta >> res_m55_mcta.log
