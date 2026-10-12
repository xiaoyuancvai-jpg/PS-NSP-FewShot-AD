
#!/bin/bash

# Python environment
export PYTHONPATH=src
export HF_HUB_OFFLINE=1

# Configurable paths and GPU
DATA_PATH="${DATA_PATH:-/path/to/mvtec-ad}"
MODEL_CKPT="${MODEL_CKPT:-/path/to/dinov2-with-registers-giant}"
GPU_ID="${GPU_ID:-0}"

# Shared experiment settings
COMMON_ARGS=(
  --dataset_name mvtec_ad
  --dataset_path "$DATA_PATH"
  --model_ckpt "$MODEL_CKPT"
  --k_shot 1
  --image_res 672
  --aug_count 30
  --pca_ev 0.99
)

echo "Starting 672px ablation experiments (30x augmentation)..."

# 1. Baseline
CUDA_VISIBLE_DEVICES="$GPU_ID" python main.py "${COMMON_ARGS[@]}" \
  --outdir ./results/ablation/exp1_baseline

# 2. Specular Filtering Only
CUDA_VISIBLE_DEVICES="$GPU_ID" python main.py "${COMMON_ARGS[@]}" \
  --use_specular_filter \
  --specular_tau 0.5 \
  --outdir ./results/ablation/exp2_spec_only

# 3. Robust PCA Only
CUDA_VISIBLE_DEVICES="$GPU_ID" python main.py "${COMMON_ARGS[@]}" \
  --use_robust_pca \
  --outdir ./results/ablation/exp3_robust_only

# 4. Full Model (Ours)
CUDA_VISIBLE_DEVICES="$GPU_ID" python main.py "${COMMON_ARGS[@]}" \
  --use_robust_pca \
  --use_specular_filter \
  --specular_tau 0.5 \
  --outdir ./results/ablation/exp4_full_model

echo "Ablation experiments completed!"
