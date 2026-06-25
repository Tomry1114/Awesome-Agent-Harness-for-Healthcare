cd ~/Medical_harness
env MH_OPENAI_MODEL=gpt-5.5 MH_JUDGE_MODEL=gpt-5.4 MH_RUBRIC_JUDGE=1 MH_GACC=1 MH_GACC_MODEL=gpt-5.4 MH_VERIFICATION_JUDGE=1 MH_GOV_JUDGE=1 /hpc2hdd/home/ce483/.conda/envs/medicalharness/bin/python runner/run_batch.py --bench PhysicianBench --max-steps 14 --fhir-base http://localhost:38080/fhir --reset-mode none --agent gpt5 --limit 10 --out results_pb_final > results_pb_final.log 2>&1
echo PB_DONE >> results_pb_final.log
