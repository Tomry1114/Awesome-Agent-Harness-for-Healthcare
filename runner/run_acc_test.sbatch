#!/bin/bash
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH -J vlm_acc
#SBATCH -o /hpc2hdd/home/ce483/Medical_harness/runner/acc_test.log
cd /hpc2hdd/home/ce483/Medical_harness
PY=~/.conda/envs/medicalharness/bin/python
IMG=benchmark/MedCTA/opencompass/data/medcta_dataset/image/image_1.jpg
Q="Based on the CT image, what type of venous thrombosis is present?"
echo "QUESTION: $Q"
echo "GOLD: Portal vein thrombosis, superior mesenteric vein thrombosis"
echo ""
$PY runner/vlm_answer.py ~/hf_models/Qwen3-VL-2B-Instruct "$IMG" "$Q" 2>/dev/null
echo ""
$PY runner/vlm_answer.py ~/hf_models/Qwen3-VL-4B-Instruct "$IMG" "$Q" 2>/dev/null
echo ""
echo "[done]"
