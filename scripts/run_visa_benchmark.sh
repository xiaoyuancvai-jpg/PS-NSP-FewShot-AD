
#!/bin/bash

# VisA dataset converted to MVTec AD format
VISA_PATH="${VISA_PATH:-/path/to/visa_mvtec_format}"
MODEL_CKPT="${MODEL_CKPT:-/path/to/dinov2-with-registers-giant}"
GPU_ID="${GPU_ID:-0}"

# ==========================================
# Task 1: VisA - Batched 0-Shot
# ==========================================
for SEED in 42 43 44 45 46; do
  echo "VisA | Batched 0-Shot | Seed: ${SEED}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH=src python main.py \
    --dataset_name mvtec_ad \
    --dataset_path "${VISA_PATH}" \
    --model_ckpt "${MODEL_CKPT}" \
    --batched_zero_shot \
    --seed "${SEED}" \
    --image_res 672 \
    --pca_ev 0.99 \
    --use_robust_pca \
    --use_specular_filter \
    --specular_tau 0.5 \
    --outdir "./results/paper_benchmark/visa_0shot_seed${SEED}"
done

# ==========================================
# Task 2: VisA - 1, 2, 4 Shot
# ==========================================
for SHOT in 1 2 4; do
  for SEED in 42 43 44 45 46; do
    echo "VisA | ${SHOT}-Shot | Seed: ${SEED}"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH=src python main.py \
      --dataset_name mvtec_ad \
      --dataset_path "${VISA_PATH}" \
      --model_ckpt "${MODEL_CKPT}" \
      --k_shot "${SHOT}" \
      --seed "${SEED}" \
      --image_res 672 \
      --aug_count 30 \
      --pca_ev 0.99 \
      --use_robust_pca \
      --use_specular_filter \
      --specular_tau 0.5 \
      --outdir "./results/paper_benchmark/visa_${SHOT}shot_seed${SEED}"
  done
done

echo "VisA benchmark completed!"
