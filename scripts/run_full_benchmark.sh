#!/bin/bash

# 定义你要跑的数据集 (这里以 MVTec-AD 为例，跑完可以换成 visa 继续)
DATASET="mvtec_ad"
DATA_PATH="${DATA_PATH:-/path/to/mvtec-ad}"
MODEL_CKPT="${MODEL_CKPT:-/path/to/dinov2-with-registers-giant}"
GPU_ID="${GPU_ID:-0}"

# 遍历 0, 1, 2, 4 shot
for SHOT in 0 1 2 4; do
  # 遍历 5 个不同的随机种子
  for SEED in 42 43 44 45 46; do
    echo "======================================================="
    echo "🚀 Starting Run: ${DATASET} | ${SHOT}-Shot | Seed: ${SEED}"
    echo "======================================================="

    # 使用可配置的 GPU 编号（默认 GPU 0）
    CUDA_VISIBLE_DEVICES=${GPU_ID} PYTHONPATH=src python main.py \
      --dataset_name ${DATASET} \
      --dataset_path ${DATA_PATH} \
      --model_ckpt "${MODEL_CKPT}" \
      --k_shot ${SHOT} \
      --seed ${SEED} \
      --image_res 672 \
      --aug_count 30 \
      --pca_ev 0.99 \
      --use_robust_pca \
      --use_specular_filter \
      --specular_tau 0.5 \
      --outdir ./results/paper_benchmark/${DATASET}_${SHOT}shot_seed${SEED}

    echo "✅ Finished Run: ${DATASET} | ${SHOT}-Shot | Seed: ${SEED}"
  done
done

echo "🎉 全量 Benchmark 跑分结束！"