cd ~/Medical_harness
env MH_OPENAI_MODEL=gpt-5.5 MH_JUDGE_MODEL=gpt-5.4 MH_RUBRIC_JUDGE=1 MH_GACC=1 MH_GACC_MODEL=gpt-5.4 MH_VERIFICATION_JUDGE=1 MH_GOV_JUDGE=1 MH_MM_JUDGE=1 MH_MM_JUDGE_MODEL=gpt-5.4 /hpc2hdd/home/ce483/.conda/envs/medicalharness/bin/python runner/run_batch.py --bench MedCTA --max-steps 12 --agent gpt5 --limit 10 --out results_mcta_final > results_mcta_final.log 2>&1
echo MCTA_DONE >> results_mcta_final.log
