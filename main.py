import os
import math
import logging
import time
from pathlib import Path

import numpy as np
import random
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
import cv2
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.decomposition import PCA
from anomalib.metrics.aupro import _AUPRO as TM_AUPRO
from src.subspacead.utils.viz import instantiate_defect_boxes
from src.subspacead.config import get_args, parse_layer_indices, parse_grouped_layers
from src.subspacead.utils.common import (
    setup_logging,
    save_config,
    min_max_norm,
)
from src.subspacead.data.datasets import get_dataset_handler
from src.subspacead.core.extractor import FeatureExtractor
from src.subspacead.core.pca import PCAModel, KernelPCAModel
from src.subspacead.post_process.scoring import calculate_anomaly_scores, post_process_map
from src.subspacead.utils.viz import save_visualization, save_overlay_for_intro, instantiate_defect_boxes
from src.subspacead.post_process.specular import (
    specular_mask_torch,
    filter_specular_anomalies,
)
from src.subspacead.core.patching import process_image_patched, get_patch_coords
from src.subspacead.data.transforms import get_augmentation_transform
from src.subspacead.core.robust_pca import augment_aware_weighted_pca


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")


def _best_f1_threshold_from_scores(y_true, y_score):
    """Return threshold maximizing F1 on validation scores."""
    y_true = np.asarray(y_true).astype(np.uint8)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.size == 0 or y_score.size == 0 or (y_true.max() == y_true.min()):
        return None, 0.0
    p, r, t = precision_recall_curve(y_true, y_score)
    if t.size == 0:
        return None, 0.0
    f1 = (2 * p[:-1] * r[:-1]) / np.clip(p[:-1] + r[:-1], 1e-12, None)
    i = int(np.nanargmax(f1))
    return float(t[i]), float(f1[i])


def _quantile_threshold_from_negatives(y_true, y_score, target_fpr=0.01):
    """
    Fallback: pick threshold so that ~target_fpr of NEGATIVES exceed it.
    y_true in {0,1}, negatives are 0. Returns None if no negatives.
    """
    y_true = np.asarray(y_true).astype(np.uint8)
    y_score = np.asarray(y_score, dtype=np.float64)
    neg = y_score[y_true == 0]
    if neg.size == 0:
        return None
    q = np.clip(1.0 - float(target_fpr), 0.0, 1.0)
    return float(np.quantile(neg, q, interpolation="linear"))


def _pick_threshold_with_fallback(y_true, y_score, target_fpr):
    """
    Try PR-optimal F1; if degenerate (single-class), fall back to negative-quantile.
    Returns (thr, how), where how ∈ {"pr", "quantile", "none"}.
    """
    thr_pr, _ = _best_f1_threshold_from_scores(y_true, y_score)
    if thr_pr is not None:
        return thr_pr, "pr"
    thr_q = _quantile_threshold_from_negatives(y_true, y_score, target_fpr)
    if thr_q is not None:
        return thr_q, "quantile"
    return None, "none"


def topk_mean(arr, frac=0.01):
    flat = arr.ravel()
    k = max(1, int(len(flat) * frac))
    idx = np.argpartition(flat, -k)[-k:]
    return float(np.mean(flat[idx]))


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"Set global random seed to {args.seed}")
    run_name = f"{args.dataset_name}_{args.agg_method}_layers{''.join(args.layers.split(','))}_res{args.image_res}_docrop{int(args.docrop)}"
    if args.patch_size:
        run_name += f"_patch{args.patch_size}"
    if args.use_kernel_pca:
        run_name += f"_kpca-{args.kernel_pca_kernel}"
    if args.use_specular_filter:
        run_name += "_spec-filt"
    if args.bg_mask_method:
        run_name += f"_mask-{args.bg_mask_method}_thr-{args.mask_threshold_method}"
        if args.mask_threshold_method == "percentile":
            run_name += f"{args.percentile_threshold}"
        if args.bg_mask_method == "dino_saliency":
            run_name += f"_L{args.dino_saliency_layer}"
    run_name += f"_score-{args.score_method}"
    run_name += f"_clahe{int(args.use_clahe)}"
    run_name += f"_dropk{args.drop_k}"
    run_name += f"_model-{args.model_ckpt.split('/')[-1]}"
    run_name += (
        f"pca_ev{args.pca_ev}" if args.pca_ev is not None else f"_pca_dim{args.pca_dim}"
    )
    run_name += f"_i-score{args.img_score_agg}"

    # Add k-shot and augmentation info to run name
    if args.k_shot is not None:
        run_name += f"_k{args.k_shot}"
        if args.aug_count > 0 and args.aug_list:
            # Create a short string for augs, e.g., "hrc"
            aug_str = "".join(sorted([a[0] for a in args.aug_list]))
            run_name += f"_aug{args.aug_count}x{aug_str}"

    args.outdir = os.path.join(args.outdir, run_name)
    os.makedirs(args.outdir, exist_ok=True)
    setup_logging(args.outdir, not args.no_log_file)
    save_config(args)

    # Augmentations
    aug_transform = None
    if args.k_shot is not None and args.aug_count > 0 and args.aug_list:
        aug_transform = get_augmentation_transform(args.aug_list, args.image_res)
        if not aug_transform.transforms:
            logging.warning(
                "Augmentation specified but no valid transforms were created. Disabling augmentations."
            )
            aug_transform = None

    # Parse layer args
    layers = parse_layer_indices(args.layers)
    grouped_layers = (
        parse_grouped_layers(args.grouped_layers) if args.agg_method == "group" else []
    )

    # Init model
    extractor = FeatureExtractor(args.model_ckpt)

    # Get dataset categories
    if args.categories:
        categories = args.categories
    else:
        categories = sorted(
            [
                f.name
                for f in Path(args.dataset_path).iterdir()
                if f.is_dir() and f.name != "split_csv"
            ]
        )

    # Main loop
    all_results = []

    # Reviewer experiment: residual-space quantitative analysis
    residual_results = []

    for category in categories:
        logging.info(f"--- Processing Category: {category} ---")

        if category in args.no_aug_categories:
            logging.warning(
                f"Disabling augmentation for {category} category"
            )
            aug_transform = None
        handler = get_dataset_handler(args.dataset_name, args.dataset_path, category)
        train_paths = handler.get_train_paths()
        val_paths = handler.get_validation_paths()
        test_paths = handler.get_test_paths()

        if args.debug_limit is not None:
            logging.warning(
                f"--- DEBUG MODE: Limiting validation and test sets to {args.debug_limit} images ---"
            )
            if val_paths:
                val_paths = val_paths[: args.debug_limit]
            if test_paths:
                test_paths = test_paths[: args.debug_limit]

        if not train_paths:
            logging.warning(f"No training images found for {category}. Skipping.")
            continue

        if args.batched_zero_shot:
            # Batched 0-shot train=test
            logging.info(
                f"--- Batched 0-Shot Mode: Fitting PCA on {len(test_paths)} test images ---"
            )
            train_paths = test_paths.copy()
            val_paths = None

        # K-shot sampling
        if args.k_shot is not None:
            if args.k_shot > len(train_paths):
                logging.warning(
                    f"Requested k_shot={args.k_shot} but only {len(train_paths)} training images available. Using all {len(train_paths)}."
                )
            else:
                logging.info(
                    f"--- K-SHOT: Randomly sampling {args.k_shot} training images ---"
                )
                random.shuffle(train_paths)
                train_paths = (
                    train_paths[: args.k_shot]
                    if args.k_shot <= len(train_paths)
                    else train_paths
                )
                for i, path in enumerate(train_paths):
                    logging.info(
                        f"  K-Shot image {i + 1}/{args.k_shot}: {Path(path).name}"
                    )

        # 1. Fit PCA Model
        if args.patch_size:
            if args.bg_mask_method == "pca_normality":
                logging.error(
                    "PCA Normality mask is not compatible with --patch_size. "
                    "Use 'dino_saliency' or no mask."
                )
                raise ValueError("Cannot use pca_normality mask with patch_size.")

            temp_img = Image.open(train_paths[0]).convert("RGB")
            temp_patch = temp_img.crop((0, 0, args.patch_size, args.patch_size))
            temp_tokens, (h_p, w_p), _ = extractor.extract_tokens(
                [temp_patch],
                args.image_res,
                layers,
                args.agg_method,
                grouped_layers,
                args.docrop,
                use_clahe=args.use_clahe,
                dino_saliency_layer=args.dino_saliency_layer,
            )
            feature_dim = temp_tokens.shape[-1]
            tokens_per_patch = h_p * w_p

            # Calculate total number of patches and tokens (with augmentations)
            total_patches = 0
            num_batches = 0
            # This multiplier accounts for the original image + N augmented images
            num_aug_multiplier = (1 + args.aug_count) if aug_transform else 1

            for path in train_paths:
                img = Image.open(path).convert("RGB")
                patch_coords = get_patch_coords(
                    img.height, img.width, args.patch_size, args.patch_overlap
                )
                total_patches += len(patch_coords) * num_aug_multiplier
                num_batches += (
                    math.ceil(len(patch_coords) / args.batch_size) * num_aug_multiplier
                )
            total_tokens = total_patches * tokens_per_patch

            logging.info(
                f"Feature dim: {feature_dim}, Tokens per patch: {tokens_per_patch}, "
                f"Base train patches: {total_patches // num_aug_multiplier}, "
                f"Total train patches (w/ aug): {total_patches}, Total train tokens: {total_tokens}"
            )

            def feature_generator_patched():
                for path in train_paths:
                    pil_img = Image.open(path).convert("RGB")

                    # Create a list of images to process: original + augmentations
                    images_to_process = [pil_img]
                    if aug_transform:
                        for _ in range(args.aug_count):
                            images_to_process.append(aug_transform(pil_img))

                    # Process each image (original + augmented)
                    for img in images_to_process:
                        patch_coords = get_patch_coords(
                            img.height,
                            img.width,
                            args.patch_size,
                            args.patch_overlap,
                        )
                        for i in range(0, len(patch_coords), args.batch_size):
                            coord_batch = patch_coords[i : i + args.batch_size]
                            patch_batch = [img.crop(c) for c in coord_batch]
                            (
                                tokens_batch,
                                _,
                                saliency_masks_batch,
                            ) = extractor.extract_tokens(
                                patch_batch,
                                args.image_res,
                                layers,
                                args.agg_method,
                                grouped_layers,
                                args.docrop,
                                use_clahe=args.use_clahe,
                                dino_saliency_layer=args.dino_saliency_layer,
                            )
                            tokens_flat = tokens_batch.reshape(-1, feature_dim)

                            if args.bg_mask_method == "dino_saliency":
                                masks_flat = saliency_masks_batch.reshape(-1)
                                try:
                                    if args.mask_threshold_method == "percentile":
                                        threshold = np.percentile(
                                            masks_flat, args.percentile_threshold * 100
                                        )
                                        foreground_tokens = tokens_flat[
                                            masks_flat >= threshold
                                        ]
                                    else:
                                        norm_mask = cv2.normalize(
                                            masks_flat,
                                            None,
                                            0,
                                            255,
                                            cv2.NORM_MINMAX,
                                            dtype=cv2.CV_8U,
                                        )
                                        _, binary_mask = cv2.threshold(
                                            norm_mask,
                                            0,
                                            255,
                                            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                                        )
                                        foreground_tokens = tokens_flat[
                                            binary_mask.flatten() > 0
                                        ]

                                    if foreground_tokens.shape[0] > 0:
                                        yield foreground_tokens
                                    else:
                                        logging.warning(
                                            "No foreground patch tokens found. Yielding all tokens."
                                        )
                                        yield tokens_flat
                                except Exception as e:
                                    logging.warning(
                                        f"Masking failed: {e}. Yielding all tokens."
                                    )
                                    yield tokens_flat
                            else:
                                yield tokens_flat

            feature_generator = feature_generator_patched

        else:
            # PCA without patching
            temp_img = Image.open(train_paths[0]).convert("RGB")
            temp_tokens, (h_p, w_p), _ = extractor.extract_tokens(
                [temp_img],
                args.image_res,
                layers,
                args.agg_method,
                grouped_layers,
                args.docrop,
                use_clahe=args.use_clahe,
                dino_saliency_layer=args.dino_saliency_layer,
            )
            feature_dim = temp_tokens.shape[-1]
            num_aug_multiplier = (1 + args.aug_count) if aug_transform else 1
            total_train_images = len(train_paths) * num_aug_multiplier
            total_tokens = total_train_images * h_p * w_p

            logging.info(
                f"Feature dim: {feature_dim}, Tokens per image: {h_p * w_p}, "
                f"Base train images: {len(train_paths)}, "
                f"Total train images (w/ aug): {total_train_images}, Total train tokens: {total_tokens}"
            )

            def feature_generator_full():
                all_imgs_to_process = []
                for path in train_paths:
                    pil_img = Image.open(path).convert("RGB")
                    all_imgs_to_process.append(pil_img)
                    if aug_transform:
                        for _ in range(args.aug_count):
                            all_imgs_to_process.append(aug_transform(pil_img))

                # Now process all_imgs_to_process in batches
                for i in range(0, len(all_imgs_to_process), args.batch_size):
                    img_batch = all_imgs_to_process[i: i + args.batch_size]

                    (
                        tokens_batch,
                        (h_p, w_p),
                        saliency_masks_batch,
                    ) = extractor.extract_tokens(
                        img_batch,
                        args.image_res,
                        layers,
                        args.agg_method,
                        grouped_layers,
                        args.docrop,
                        use_clahe=args.use_clahe,
                        dino_saliency_layer=args.dino_saliency_layer,
                    )
                    tokens_flat = tokens_batch.reshape(-1, feature_dim)
                    keep_mask = np.ones(tokens_flat.shape[0], dtype=bool)

                    # [恢复原状]：只保留背景 Saliency 过滤，彻底移除愚蠢的训练期高光拦截！
                    if args.bg_mask_method == "dino_saliency":
                        masks_flat = saliency_masks_batch.reshape(-1)
                        try:
                            if args.mask_threshold_method == "percentile":
                                threshold = np.percentile(masks_flat, args.percentile_threshold * 100)
                                bg_keep = (masks_flat >= threshold)
                            else:
                                norm_mask = cv2.normalize(masks_flat, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                                _, binary_mask = cv2.threshold(norm_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                bg_keep = (binary_mask.flatten() > 0)

                            keep_mask = keep_mask & bg_keep
                        except Exception as e:
                            logging.warning(f"Background Masking failed: {e}. Skipping BG mask.")

                    valid_tokens = tokens_flat[keep_mask]

                    if valid_tokens.shape[0] > 0:
                        yield valid_tokens
                    else:
                        yield tokens_flat



            num_batches = math.ceil(total_train_images / args.batch_size)
            feature_generator = feature_generator_full

        if args.use_kernel_pca:
            if args.bg_mask_method == "pca_normality":
                logging.error(
                    "PCA Normality mask is not compatible with Kernel PCA. "
                    "Use 'dino_saliency' or no mask."
                )
                raise ValueError("Cannot use pca_normality mask with use_kernel_pca.")

            logging.info("Collecting all features for Kernel PCA...")
            all_train_tokens = np.concatenate(
                list(
                    tqdm(
                        feature_generator(),
                        desc="Feature Collection",
                        total=num_batches,
                    )
                )
            )
            pca_model = KernelPCAModel(
                k=args.pca_dim,
                kernel=args.kernel_pca_kernel,
                gamma=args.kernel_pca_gamma,
            )
            pca_params = pca_model.fit(all_train_tokens)
        else:
            USE_ROBUST_PCA = getattr(args, 'use_robust_pca', False)

            if USE_ROBUST_PCA:
                logging.info(">>> [OURS] Running Robust Residual-Aware Weighted PCA...")
                # 1. 收集全量特征
                all_train_tokens = np.concatenate(
                    list(
                        tqdm(
                            feature_generator(),
                            desc="Collecting Train Tokens for Robust PCA",
                            total=num_batches,
                            leave=False
                        )
                    )
                )

                # ================= 新增：空间换时间/精度的下采样防爆机制 =================
                MAX_TOKENS = 100000  # 如果你的显卡还是爆显存，把这个值下调到 50000
                if all_train_tokens.shape[0] > MAX_TOKENS:
                    logging.warning(
                        f"Tokens 数量 ({all_train_tokens.shape[0]}) 逼近显存极限，随机采样 {MAX_TOKENS} 个特征点...")
                    # 随机采样，打乱顺序，提取子集
                    indices = np.random.choice(all_train_tokens.shape[0], MAX_TOKENS, replace=False)
                    all_train_tokens = all_train_tokens[indices]
                # ====================================================================

                # 2. 转为 Tensor 并送入 GPU
                tokens_tensor = torch.from_numpy(all_train_tokens).to(DEVICE, dtype=torch.float32)
                # 3. 运行你的核心算法
                V_final, w_mean, lambdas = augment_aware_weighted_pca(
                    tokens_tensor,
                    pca_ev=args.pca_ev if args.pca_ev is not None else 0.99,
                    alpha=2.0,
                    w_min=0.01
                )

                # 4. 封装回官方所需的格式
                pca_params = {
                    "mu": w_mean,
                    "components": V_final,
                    "singular_values": np.sqrt(lambdas * (total_tokens - 1)),
                    "whiten": args.whiten,
                    "k": V_final.shape[1],
                }
                logging.info(f"Robust Weighted PCA finished. Retained components: {V_final.shape[1]}")
            else:
                # 官方原版 Baseline 逻辑
                pca_model = PCAModel(k=args.pca_dim, ev=args.pca_ev, whiten=args.whiten)
                pca_params = pca_model.fit(
                    feature_generator,
                    feature_dim,
                    total_tokens,
                    num_batches,
                )
            try:
                print(f">> [Hook] Extracting train features for {category} visualization...")
                train_feats_all = np.concatenate(list(feature_generator()))
                # 加入 category 防止被其他类覆盖
                save_dir = os.path.join(args.outdir, category, "manifold_features")
                os.makedirs(save_dir, exist_ok=True)
                aug_suffix = f"AUG_{args.aug_count}" if args.aug_count > 0 else "NO_AUG"
                np.save(os.path.join(save_dir, f"{aug_suffix}_train.npy"), train_feats_all)
            except Exception as e:
                print(f"Failed to save train features for {category}: {e}")

        # 2. Determine PR-optimal F1 thresholds (if validation set exists)
        if val_paths:
            logging.info(
                f"Collecting validation stats on {len(val_paths)} images for PR-optimal F1 thresholds..."
            )
            val_img_scores, val_img_labels = [], []
            val_px_scores_normalized, val_px_gts = [], []
            val_iter = tqdm(val_paths, desc="Validating")
            for i in range(0, len(val_paths), args.batch_size):
                path_batch = val_paths[i : i + args.batch_size]
                pil_imgs = [Image.open(p).convert("RGB") for p in path_batch]
                # --- [NEW] 施加受控空间旋转 ---
                if getattr(args, "test_rotation_angle", 0.0) != 0.0:
                    print(f"\n[DEBUG] 成功触发！正在将测试图像旋转 {args.test_rotation_angle} 度...")
                    # 填充为白色 (255, 255, 255)，与 MVTec 原始背景完美融合
                    pil_imgs = [img.rotate(args.test_rotation_angle, resample=Image.BILINEAR, fillcolor=(255, 255, 255))
                                for img in pil_imgs]
                # ------------------------------
                is_anomaly_batch = [
                    "good" not in p and "Normal" not in p for p in path_batch
                ]

                if args.patch_size:
                    anomaly_maps_batch, _ = process_image_patched(
                        pil_imgs,
                        extractor,
                        pca_params,
                        args,
                        DEVICE,
                        h_p,
                        w_p,
                        feature_dim,
                    )
                    for j, anomaly_map_final in enumerate(anomaly_maps_batch):
                        if args.img_score_agg == "max":
                            img_score = float(np.max(anomaly_map_final))
                        elif args.img_score_agg == "p99":
                            img_score = float(np.percentile(anomaly_map_final, 99))
                        elif args.img_score_agg == "mtop5":
                            img_score = float(
                                np.mean(np.sort(anomaly_map_final.flatten())[-5:])
                            )
                        elif args.img_score_agg == "mtop1p":
                            img_score = topk_mean(anomaly_map_final, frac=0.01)
                        else:
                            img_score = float(np.mean(anomaly_map_final))
                        val_img_scores.append(img_score)
                        val_img_labels.append(1 if is_anomaly_batch[j] else 0)

                        # --- PIXEL METRICS (AUPRO, P-F1) ---
                        anomaly_map_normalized = min_max_norm(anomaly_map_final)
                        H, W = anomaly_map_normalized.shape
                        gt_mask = handler.get_ground_truth_mask(
                            path_batch[j], pil_imgs[j].size
                        )
                        gt_mask = (
                            np.array(
                                Image.fromarray(
                                    (gt_mask.astype(np.uint8) * 255)
                                ).resize((W, H), resample=Image.NEAREST)
                            )
                            > 127
                        )
                        val_px_gts.extend(gt_mask.flatten().astype(np.uint8))
                        val_px_scores_normalized.extend(
                            anomaly_map_normalized.flatten().astype(np.float32)
                        )

                else:
                    (
                        tokens,
                        (h_p, w_p),
                        saliency_masks_batch,
                    ) = extractor.extract_tokens(
                        pil_imgs,
                        args.image_res,
                        layers,
                        args.agg_method,
                        grouped_layers,
                        args.docrop,
                        use_clahe=args.use_clahe,
                        dino_saliency_layer=args.dino_saliency_layer,
                    )
                    b, _, _, c = tokens.shape
                    tokens_reshaped = tokens.reshape(b * h_p * w_p, c)

                    scores = calculate_anomaly_scores(
                        tokens_reshaped,
                        pca_params,
                        args.score_method,
                        args.drop_k,
                    )
                    anomaly_maps = scores.reshape(b, h_p, w_p)

                    # Apply masking to validation
                    if args.bg_mask_method == "dino_saliency":
                        background_mask = np.zeros_like(anomaly_maps, dtype=bool)
                        for j in range(b):
                            saliency_map = saliency_masks_batch[j]
                            try:
                                if args.mask_threshold_method == "percentile":
                                    threshold = np.percentile(
                                        saliency_map, args.percentile_threshold * 100
                                    )
                                    background_mask[j] = saliency_map < threshold
                                else:  # otsu
                                    norm_mask = cv2.normalize(
                                        saliency_map,
                                        None,
                                        0,
                                        255,
                                        cv2.NORM_MINMAX,
                                        dtype=cv2.CV_8U,
                                    )
                                    _, binary_mask = cv2.threshold(
                                        norm_mask,
                                        0,
                                        255,
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                                    )
                                    background_mask[j] = binary_mask == 0
                            except Exception as e:
                                logging.warning(
                                    f"Saliency mask failed for val image {j}: {e}. Skipping mask."
                                )
                        anomaly_maps[background_mask] = 0.0

                    elif args.bg_mask_method == "pca_normality":
                        # AnomalyDINO PCA mask
                        threshold = 10.0
                        kernel_size = 3
                        border = 0.2
                        grid_size = (h_p, w_p)
                        kernel = np.ones(
                            (kernel_size, kernel_size), np.uint8
                        )  # Pre-define kernel

                        background_mask_batch = np.zeros_like(anomaly_maps, dtype=bool)

                        for j in range(b):
                            img_features = tokens[j].reshape(-1, c)

                            try:
                                pca = PCA(n_components=1, svd_solver="randomized")
                                first_pc = pca.fit_transform(
                                    img_features.astype(np.float32)
                                )

                                mask = first_pc > threshold
                                mask_2d = mask.reshape(grid_size)
                                h_start, h_end = int(grid_size[0] * border), int(
                                    grid_size[0] * (1 - border)
                                )
                                w_start, w_end = int(grid_size[1] * border), int(
                                    grid_size[1] * (1 - border)
                                )
                                m = mask_2d[h_start:h_end, w_start:w_end]

                                if m.sum() <= m.size * 0.35:
                                    mask = -first_pc > threshold
                                    mask_2d = mask.reshape(grid_size)

                                # Post-process foreground mask
                                mask_processed = cv2.dilate(
                                    mask_2d.astype(np.uint8), kernel
                                ).astype(bool)
                                mask_processed = cv2.morphologyEx(
                                    mask_processed.astype(np.uint8),
                                    cv2.MORPH_CLOSE,
                                    kernel,
                                ).astype(bool)

                                # Invert the foreground mask to get the background mask
                                background_mask_batch[j] = ~mask_processed

                            except Exception as e:
                                logging.warning(
                                    f"PCA mask failed for val image {j}: {e}. Skipping mask."
                                )

                        anomaly_maps[background_mask_batch] = 0.0
                    for j in range(anomaly_maps.shape[0]):
                        anomaly_map_final = post_process_map(
                            anomaly_maps[j], args.image_res,
                            input_img = cv2.cvtColor(np.array(pil_imgs[j]), cv2.COLOR_RGB2BGR)
                        )

                        if args.use_specular_filter:
                            img_tensor = (
                                TF.to_tensor(pil_imgs[j]).unsqueeze(0).to(DEVICE)
                            )
                            _, _, conf = specular_mask_torch(
                                img_tensor, tau=args.specular_tau
                            )
                            conf = torch.nn.functional.interpolate(
                                conf,
                                size=anomaly_map_final.shape,
                                mode="bilinear",
                                align_corners=False,
                            )
                            conf_map = conf.squeeze().cpu().numpy()
                            anomaly_map_final = (
                                filter_specular_anomalies(anomaly_map_final, conf_map)
                                .cpu()
                                .numpy()
                            )
                        if args.img_score_agg == "max":
                            img_score = float(np.max(anomaly_map_final))
                        elif args.img_score_agg == "p99":
                            img_score = float(np.percentile(anomaly_map_final, 99))
                        elif args.img_score_agg == "mtop5":
                            img_score = float(
                                np.mean(np.sort(anomaly_map_final.flatten())[-5:])
                            )
                        elif args.img_score_agg == "mtop1p":
                            img_score = topk_mean(anomaly_map_final, frac=0.01)
                        else:
                            img_score = float(np.mean(anomaly_map_final))
                        val_img_scores.append(img_score)
                        val_img_labels.append(1 if is_anomaly_batch[j] else 0)
                        anomaly_map_normalized = min_max_norm(anomaly_map_final)
                        H, W = anomaly_map_normalized.shape
                        gt_path_str = handler.get_ground_truth_path(path_batch[j])

                        if not gt_path_str or not os.path.exists(gt_path_str):
                            gt_mask = np.zeros((H, W), dtype=np.uint8)
                        else:
                            gt_mask_pil = Image.open(gt_path_str).convert("L")
                            # --- [NEW] 同步旋转掩码，确保评估指标对齐 ---
                            if getattr(args, "test_rotation_angle", 0.0) != 0.0:
                                gt_mask_pil = gt_mask_pil.rotate(args.test_rotation_angle, resample=Image.NEAREST)
                            # ---------------------------------------------

                            if args.docrop:
                                resize_res = int(args.image_res / 0.875)
                                gt_mask_pil = TF.resize(
                                    gt_mask_pil,
                                    (resize_res, resize_res),
                                    interpolation=TF.InterpolationMode.NEAREST,
                                )
                                gt_mask_pil = TF.center_crop(
                                    gt_mask_pil, (args.image_res, args.image_res)
                                )

                            gt_mask_pil = TF.resize(
                                gt_mask_pil,
                                (H, W),
                                interpolation=TF.InterpolationMode.NEAREST,
                            )
                            gt_mask = (np.array(gt_mask_pil) > 0).astype(np.uint8)

                        val_px_gts.extend(gt_mask.flatten().astype(np.uint8))
                        val_px_scores_normalized.extend(
                            anomaly_map_normalized.flatten().astype(np.float32)
                        )
                val_iter.update(len(path_batch))

            target_img_fpr = getattr(args, "target_img_fpr", 0.05)
            target_px_fpr = getattr(args, "target_px_fpr", 0.05)

            # Threshold for I-F1 (using raw image scores)
            thr_img, how_img = _pick_threshold_with_fallback(
                val_img_labels, val_img_scores, target_img_fpr
            )
            # Threshold for P-F1 (using per-image normalized pixel scores)
            val_px_scores_mm = np.array(val_px_scores_normalized)
            thr_px, how_px = _pick_threshold_with_fallback(
                val_px_gts, val_px_scores_mm, target_px_fpr
            )

            if how_img == "none":
                logging.warning(
                    "Validation image threshold degenerate and no negatives: image F1 will be NaN."
                )
            if how_px == "none":
                logging.warning(
                    "Validation pixel threshold degenerate and no negatives: pixel F1 will be NaN."
                )

            logging.info(
                f"Chosen thresholds — Image: {thr_img if thr_img is not None else float('nan'):.6g} "
                f"({how_img}), Pixel: {thr_px if thr_px is not None else float('nan'):.6g} ({how_px})"
            )

        else:
            logging.warning("No validation set found. F1 scores will be N/A.")
            thr_img, thr_px = None, None

        # Warm up for timing
        if test_paths:
            logging.info("Performing warm-up inference run...")
            try:
                # Use the first test image for the warm-up
                dummy_img = [Image.open(test_paths[0]).convert("RGB")]

                if args.patch_size:
                    # Warm-up the patch pipeline
                    _ = process_image_patched(
                        dummy_img,
                        extractor,
                        pca_params,
                        args,
                        DEVICE,
                        h_p,
                        w_p,
                        feature_dim,
                    )
                else:
                    # Warm-up the full-image pipeline
                    _tokens, (_h, _w), _saliency = extractor.extract_tokens(
                        dummy_img,
                        args.image_res,
                        layers,
                        args.agg_method,
                        grouped_layers,
                        args.docrop,
                        use_clahe=args.use_clahe,
                        dino_saliency_layer=args.dino_saliency_layer,
                    )
                    # A minimal version of the scoring
                    _scores = calculate_anomaly_scores(
                        _tokens.reshape(-1, _tokens.shape[-1]),
                        pca_params,
                        args.score_method,
                        args.drop_k,
                    )
                    if args.use_specular_filter and torch.cuda.is_available():
                        _ = filter_specular_anomalies(
                            torch.from_numpy(_scores).to(DEVICE),
                            torch.zeros_like(torch.from_numpy(_scores)).to(DEVICE),
                        )

                torch.cuda.synchronize(DEVICE)
                logging.info("Warm-up complete.")
            except Exception as e:
                logging.warning(
                    f"Warm-up run failed: {e}. First timed run may be slow."
                )

        # 3. Evaluate on Test Set
        logging.info(f"Evaluating on {len(test_paths)} test images...")
        img_true, img_pred_f1 = [], []
        img_pred_auroc = []
        px_true_all = []
        px_pred_all_auroc = []
        px_pred_all_normalized = []
        anomalous_gt_masks = []
        anomalous_anomaly_maps = []
        vis_saved_count = 0
        all_inference_times = []

        logging.info("Number of test images: {}".format(len(test_paths)))
        test_normal_tokens_list = []
        test_anomaly_tokens_list = []

        # Reviewer experiment: token-level reconstruction residual statistics
        residual_normal_list = []
        residual_anomaly_list = []
        nre_normal_list = []
        nre_anomaly_list = []

        test_iter = tqdm(test_paths, desc=f"Testing {category}")
        for i in range(0, len(test_paths), args.batch_size):
            path_batch = test_paths[i : i + args.batch_size]
            pil_imgs = [Image.open(p).convert("RGB") for p in path_batch]
            test_rotation_angle = getattr(args, "test_rotation_angle", 0.0)
            if test_rotation_angle != 0.0:
                if i == 0:  # 只在第一个 batch 打印一次，防止刷屏
                    logging.info(f">>> [DEBUG] FORCE ROTATING TEST IMAGES BY {test_rotation_angle} DEGREES! <<<")
                # 使用双线性插值旋转原图，避免引入过多伪影
                pil_imgs = [img.rotate(test_rotation_angle, resample=Image.BILINEAR, fillcolor=(255, 255, 255)) for img in pil_imgs]
            is_anomaly_batch = [
                "good" not in str(p) and "Normal" not in str(p) for p in path_batch
            ]
            torch.cuda.synchronize(DEVICE)
            start_time = time.perf_counter()

            final_anomaly_maps_for_batch = []
            saliency_maps_for_viz_batch = []

            if args.patch_size:
                (
                    anomaly_maps_batch,
                    saliency_maps_batch,
                ) = process_image_patched(
                    pil_imgs, extractor, pca_params, args, DEVICE, h_p, w_p, feature_dim
                )

                saliency_maps_for_viz_batch = saliency_maps_batch

                for j, anomaly_map_pre_specular in enumerate(anomaly_maps_batch):
                    anomaly_map_final = anomaly_map_pre_specular
                    if args.use_specular_filter:
                        img_tensor = TF.to_tensor(pil_imgs[j]).unsqueeze(0).to(DEVICE)
                        _, _, conf = specular_mask_torch(
                            img_tensor, tau=args.specular_tau
                        )
                        conf = torch.nn.functional.interpolate(
                            conf,
                            size=anomaly_map_pre_specular.shape,
                            mode="bilinear",
                            align_corners=False,
                        )
                        conf_map = conf.squeeze().cpu().numpy()
                        anomaly_map_final = (
                            filter_specular_anomalies(
                                anomaly_map_pre_specular, conf_map
                            )
                            .cpu()
                            .numpy()
                        )
                    final_anomaly_maps_for_batch.append(anomaly_map_final)

            else:
                # Step 1: Feature Extraction
                (
                    tokens,
                    (h_p, w_p),
                    saliency_masks_batch,
                ) = extractor.extract_tokens(
                    pil_imgs,
                    args.image_res,
                    layers,
                    args.agg_method,
                    grouped_layers,
                    args.docrop,
                    use_clahe=args.use_clahe,
                    dino_saliency_layer=args.dino_saliency_layer,
                )
                b, _, _, c = tokens.shape
                tokens_reshaped = tokens.reshape(b * h_p * w_p, c)
                if not args.patch_size:  # 删除了 category == 'screw' 的限制

                    for _idx_in_batch in range(b):
                        _img_tokens = tokens[_idx_in_batch].reshape(-1, c).cpu().numpy() if torch.is_tensor(tokens) else \
                            tokens[_idx_in_batch].reshape(-1, c)
                        # 每张图随机采样 20 个 token (保证文件不会太大，画图时点也不会挤成一团)
                        np.random.seed(42)
                        _sampled_tokens = _img_tokens[
                            np.random.choice(_img_tokens.shape[0], min(20, _img_tokens.shape[0]), replace=False)]
                        if is_anomaly_batch[_idx_in_batch]:
                            test_anomaly_tokens_list.append(_sampled_tokens)
                        else:
                            test_normal_tokens_list.append(_sampled_tokens)
                # Step 2: Anomaly Scoring
                scores = calculate_anomaly_scores(
                    tokens_reshaped,
                    pca_params,
                    args.score_method,
                    args.drop_k,
                )
                anomaly_maps = scores.reshape(b, h_p, w_p)

                # ============================================================
                # Reviewer experiment: residual-space quantitative analysis
                # This block ONLY records statistics and does not modify
                # anomaly_maps or the original benchmark evaluation pipeline.
                # ============================================================
                if args.score_method == "reconstruction":
                    scores_np = np.asarray(scores, dtype=np.float64)

                    # calculate_anomaly_scores(reconstruction) returns squared
                    # reconstruction error; take sqrt for L2 residual statistics.
                    residual_maps = np.sqrt(
                        np.clip(scores_np, 0.0, None)
                    ).reshape(b, h_p, w_p)

                    # Normalized residual energy controls for different retained
                    # PCA dimensionalities under the same EV threshold.
                    mu_for_residual = pca_params["mu"]
                    if torch.is_tensor(mu_for_residual):
                        mu_for_residual = (
                            mu_for_residual.detach().cpu().numpy()
                        )
                    mu_for_residual = np.asarray(
                        mu_for_residual, dtype=np.float64
                    ).reshape(1, -1)

                    tokens_for_residual = tokens_reshaped
                    if torch.is_tensor(tokens_for_residual):
                        tokens_for_residual = (
                            tokens_for_residual.detach().cpu().numpy()
                        )
                    tokens_for_residual = np.asarray(
                        tokens_for_residual, dtype=np.float64
                    )

                    centered_energy = np.sum(
                        (tokens_for_residual - mu_for_residual) ** 2,
                        axis=1,
                    )
                    nre_maps = (
                        scores_np / (centered_energy + 1e-12)
                    ).reshape(b, h_p, w_p)

                    for _rj in range(b):
                        _path = path_batch[_rj]
                        _gt_path = handler.get_ground_truth_path(_path)

                        if not _gt_path or not os.path.exists(_gt_path):
                            _gt_token = np.zeros(
                                (h_p, w_p), dtype=np.uint8
                            )
                        else:
                            _gt_pil = Image.open(_gt_path).convert("L")

                            _rot = getattr(
                                args, "test_rotation_angle", 0.0
                            )
                            if _rot != 0.0:
                                _gt_pil = _gt_pil.rotate(
                                    _rot,
                                    resample=Image.NEAREST,
                                )

                            if args.docrop:
                                _resize_res = int(
                                    args.image_res / 0.875
                                )
                                _gt_pil = TF.resize(
                                    _gt_pil,
                                    (_resize_res, _resize_res),
                                    interpolation=TF.InterpolationMode.NEAREST,
                                )
                                _gt_pil = TF.center_crop(
                                    _gt_pil,
                                    (args.image_res, args.image_res),
                                )

                            # Map the pixel GT to the DINO token grid.
                            # For the 672px / ViT-14 setting this becomes
                            # 48x48 tokens and uses max-overlap pooling.
                            _gt_pil = TF.resize(
                                _gt_pil,
                                (args.image_res, args.image_res),
                                interpolation=TF.InterpolationMode.NEAREST,
                            )
                            _gt_np = (
                                np.array(_gt_pil) > 0
                            ).astype(np.uint8)

                            if (
                                args.image_res % h_p == 0
                                and args.image_res % w_p == 0
                            ):
                                _kh = args.image_res // h_p
                                _kw = args.image_res // w_p
                                _gt_token = _gt_np.reshape(
                                    h_p, _kh, w_p, _kw
                                ).max(axis=(1, 3)).astype(np.uint8)
                            else:
                                _gt_token = cv2.resize(
                                    _gt_np,
                                    (w_p, h_p),
                                    interpolation=cv2.INTER_NEAREST,
                                ).astype(np.uint8)

                        _rmap = residual_maps[_rj]
                        _nmap = nre_maps[_rj]

                        _normal_mask = (_gt_token == 0)
                        _anomaly_mask = (_gt_token == 1)

                        if np.any(_normal_mask):
                            residual_normal_list.append(
                                _rmap[_normal_mask].astype(np.float32)
                            )
                            nre_normal_list.append(
                                _nmap[_normal_mask].astype(np.float32)
                            )

                        if np.any(_anomaly_mask):
                            residual_anomaly_list.append(
                                _rmap[_anomaly_mask].astype(np.float32)
                            )
                            nre_anomaly_list.append(
                                _nmap[_anomaly_mask].astype(np.float32)
                            )
                # ============================================================

                # Step 3: Masking Strategy
                mask_for_viz = None
                background_mask = np.zeros_like(anomaly_maps, dtype=bool)

                if args.bg_mask_method == "dino_saliency":
                    mask_for_viz = saliency_masks_batch
                    for j in range(b):
                        saliency_map = saliency_masks_batch[j]
                        try:
                            if args.mask_threshold_method == "percentile":
                                threshold = np.percentile(
                                    saliency_map, args.percentile_threshold * 100
                                )
                                background_mask[j] = saliency_map < threshold
                            else:  # otsu
                                norm_mask = cv2.normalize(
                                    saliency_map,
                                    None,
                                    0,
                                    255,
                                    cv2.NORM_MINMAX,
                                    dtype=cv2.CV_8U,
                                )
                                _, binary_mask = cv2.threshold(
                                    norm_mask,
                                    0,
                                    255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                                )
                                background_mask[j] = binary_mask == 0
                        except Exception as e:
                            logging.warning(
                                f"Saliency mask failed for test image {j}: {e}. Skipping mask."
                            )

                elif args.bg_mask_method == "pca_normality":
                    threshold = 10.0
                    kernel_size = 3
                    border = 0.2
                    grid_size = (h_p, w_p)
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    mask_for_viz = np.zeros_like(anomaly_maps)

                    for j in range(b):
                        img_features = tokens[j].reshape(-1, c)
                        try:
                            pca = PCA(n_components=1, svd_solver="randomized")
                            first_pc = pca.fit_transform(
                                img_features.astype(np.float32)
                            )
                            mask = first_pc > threshold
                            mask_2d = mask.reshape(grid_size)

                            h_start, h_end = int(grid_size[0] * border), int(
                                grid_size[0] * (1 - border)
                            )
                            w_start, w_end = int(grid_size[1] * border), int(
                                grid_size[1] * (1 - border)
                            )
                            m = mask_2d[h_start:h_end, w_start:w_end]

                            if m.sum() <= m.size * 0.35:
                                mask = -first_pc > threshold
                                mask_2d = mask.reshape(grid_size)

                            mask_processed = cv2.dilate(
                                mask_2d.astype(np.uint8), kernel
                            ).astype(bool)
                            mask_processed = cv2.morphologyEx(
                                mask_processed.astype(np.uint8), cv2.MORPH_CLOSE, kernel
                            ).astype(bool)

                            background_mask[j] = ~mask_processed
                            mask_for_viz[j] = mask_processed.astype(np.float32)
                        except Exception as e:
                            logging.warning(
                                f"PCA mask failed for test image {j}: {e}. Skipping mask."
                            )

                anomaly_maps[background_mask] = 0.0

                saliency_maps_for_viz_batch = mask_for_viz

                for j in range(anomaly_maps.shape[0]):
                    pil_img = pil_imgs[j]
                    anomaly_map_pre_specular = post_process_map(
                        anomaly_maps[j], args.image_res,
                        input_img=cv2.cvtColor(np.array(pil_imgs[j]), cv2.COLOR_RGB2BGR)
                    )
                    anomaly_map_final = anomaly_map_pre_specular
                    if args.use_specular_filter:
                        img_tensor = TF.to_tensor(pil_imgs[j]).unsqueeze(0).to(DEVICE)
                        _, _, conf = specular_mask_torch(
                            img_tensor, tau=args.specular_tau
                        )
                        conf = torch.nn.functional.interpolate(
                            conf,
                            size=anomaly_map_final.shape,
                            mode="bilinear",
                            align_corners=False,
                        )
                        conf_map = conf.squeeze().cpu().numpy()
                        anomaly_map_final = (
                            filter_specular_anomalies(anomaly_map_final, conf_map)
                            .cpu()
                            .numpy()
                        )
                    final_anomaly_maps_for_batch.append(anomaly_map_final)

            # End timing
            torch.cuda.synchronize(DEVICE)
            end_time = time.perf_counter()
            all_inference_times.append(end_time - start_time)

            for j, anomaly_map_final in enumerate(final_anomaly_maps_for_batch):
                is_anomaly = is_anomaly_batch[j]
                path = path_batch[j]
                pil_img = pil_imgs[j]
                if args.img_score_agg == "max":
                    img_score = np.max(anomaly_map_final)
                elif args.img_score_agg == "p99":
                    img_score = np.percentile(anomaly_map_final, 99)
                elif args.img_score_agg == "mtop5":
                    img_score = np.mean(np.sort(anomaly_map_final.flatten())[-5:])
                elif args.img_score_agg == "mtop1p":
                    img_score = topk_mean(anomaly_map_final, frac=0.01)
                else:
                    img_score = np.mean(anomaly_map_final)

                img_true.append(1 if is_anomaly else 0)
                img_pred_auroc.append(float(img_score))
                if thr_img is not None:
                    img_pred_f1.append(1 if img_score >= thr_img else 0)

                anomaly_map_normalized = min_max_norm(anomaly_map_final)
                H, W = anomaly_map_normalized.shape
                # ================= 替换开始 =================
                # 去掉了 if category == 'screw': 的限制，所有类别火力全开！

                # 确定保存文件夹 (加入了 category 变量，防止15个类别混在一起)
                aug_suffix = f"AUG_{args.aug_count}" if args.aug_count > 0 else "NO_AUG"
                box_dir = os.path.join(args.outdir, category, f"bounding_boxes_{aug_suffix}")
                os.makedirs(box_dir, exist_ok=True)

                # 将 PIL 格式原图转为 CV2 可以画图的 BGR 格式，并统一大小
                _orig_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                _orig_img = cv2.resize(_orig_img, (W, H))

                # 调用画框模块
                boxed_img, d_count = instantiate_defect_boxes(_orig_img, anomaly_map_normalized, threshold=0.5,
                                                              min_area=50)

                # 加上前缀保存 (norm是正常图，anom是异常缺陷图)
                prefix = "anom" if is_anomaly else "norm"

                # 提取上一级文件夹的名字（如 manipulated_front）拼接到文件名里
                defect_type = Path(path).parent.name

                save_name = f"{prefix}_{defect_type}_{Path(path).stem}_detected.png"
                cv2.imwrite(os.path.join(box_dir, save_name), boxed_img)

                npy_save_name = f"{prefix}_{defect_type}_{Path(path).stem}_map.npy"
                np.save(os.path.join(box_dir, npy_save_name), anomaly_map_final)  # 保持保存绝对残差
                # ================= 替换结束 =================
                # ==========================================
                gt_path_str = handler.get_ground_truth_path(path)
                if not gt_path_str or not os.path.exists(gt_path_str):
                    gt_mask = np.zeros((H, W), dtype=np.uint8)
                else:
                    gt_mask_pil = Image.open(gt_path_str).convert("L")
                    test_rotation_angle = getattr(args, "test_rotation_angle", 0.0)
                    if test_rotation_angle != 0.0:
                        # 掩码必须用最近邻插值，保证 0/1 标签不被模糊化
                        gt_mask_pil = gt_mask_pil.rotate(test_rotation_angle, resample=Image.NEAREST)
                    if args.docrop:
                        resize_res = int(args.image_res / 0.875)
                        gt_mask_pil = TF.resize(
                            gt_mask_pil,
                            (resize_res, resize_res),
                            interpolation=TF.InterpolationMode.NEAREST,
                        )
                        gt_mask_pil = TF.center_crop(
                            gt_mask_pil, (args.image_res, args.image_res)
                        )
                    gt_mask_pil = TF.resize(
                        gt_mask_pil,
                        (H, W),
                        interpolation=TF.InterpolationMode.NEAREST,
                    )
                    gt_mask = (np.array(gt_mask_pil) > 0).astype(np.uint8)

                px_true_all.extend(gt_mask.flatten().astype(np.uint8))
                px_pred_all_auroc.extend(anomaly_map_final.flatten().astype(np.float32))
                px_pred_all_normalized.extend(
                    anomaly_map_normalized.flatten().astype(np.float32)
                )

                if is_anomaly:
                    anomalous_gt_masks.append(gt_mask)
                    anomalous_anomaly_maps.append(anomaly_map_normalized)

                    if args.save_intro_overlays:
                        vis_img = pil_img
                        save_overlay_for_intro(
                            path,
                            vis_img,
                            anomaly_map_normalized,
                            args.outdir,
                            category,
                        )
                    if vis_saved_count < args.vis_count:
                        vis_img = pil_img
                        if args.docrop and not args.patch_size:
                            resize_res = int(args.image_res / 0.875)
                            vis_img = TF.resize(
                                vis_img,
                                (resize_res, resize_res),
                                interpolation=TF.InterpolationMode.BICUBIC,
                            )
                            vis_img = TF.center_crop(
                                vis_img, (args.image_res, args.image_res)
                            )
                        saliency_map_for_viz = None
                        raw_mask_map = None
                        if saliency_maps_for_viz_batch is not None:
                            raw_mask_map = saliency_maps_for_viz_batch[j]

                        if raw_mask_map is not None:
                            try:
                                if args.bg_mask_method == "pca_normality":
                                    binary_mask = raw_mask_map

                                elif args.bg_mask_method == "dino_saliency":
                                    if args.mask_threshold_method == "percentile":
                                        threshold_val = np.percentile(
                                            raw_mask_map,
                                            args.percentile_threshold * 100,
                                        )
                                        binary_mask = (
                                            raw_mask_map >= threshold_val
                                        ).astype(np.float32)
                                    else:  # otsu
                                        norm_mask = cv2.normalize(
                                            raw_mask_map,
                                            None,
                                            0,
                                            255,
                                            cv2.NORM_MINMAX,
                                            dtype=cv2.CV_8U,
                                        )
                                        _, binary_mask_u8 = cv2.threshold(
                                            norm_mask,
                                            0,
                                            255,
                                            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                                        )
                                        binary_mask = (binary_mask_u8 > 0).astype(
                                            np.float32
                                        )
                                saliency_map_for_viz = post_process_map(
                                    binary_mask,
                                    anomaly_map_normalized.shape,
                                    blur=False,
                                )
                            except Exception as e:
                                logging.warning(
                                    f"Saliency mask processing failed for visualization: {e}."
                                )

                        save_visualization(
                            path,
                            vis_img,
                            gt_mask,
                            anomaly_map_normalized,
                            args.outdir,
                            category,
                            vis_saved_count,
                            saliency_mask=saliency_map_for_viz,
                        )
                        vis_saved_count += 1

            test_iter.update(len(path_batch))
        if all_inference_times:
            times_arr = np.array(all_inference_times)
            total_images_processed = len(test_paths)
            total_time = np.sum(times_arr)

            avg_time_per_image = total_time / total_images_processed
            images_per_second = 1.0 / avg_time_per_image

            logging.info(f"--- Timing Results for {category} ---")
            logging.info(f"Total test images: {total_images_processed}")
            logging.info(
                f"Batch size: {args.batch_size} (Processed {len(all_inference_times)} batches)"
            )
            logging.info(f"Total inference time: {total_time:.4f} s")
            logging.info(f"Avg. time per image: {avg_time_per_image:.6f} s")
            logging.info(f"Images per second (FPS): {images_per_second:.2f}")

            # Report batch stats
            if len(times_arr) > 1:
                times_arr_stats = times_arr[1:]
                logging.info(
                    f"Avg. time per batch (excl. 1st): {np.mean(times_arr_stats):.6f} s"
                )
                logging.info(
                    f"Median time per batch (excl. 1st): {np.median(times_arr_stats):.6f} s"
                )
            else:
                logging.info(f"Avg. time per batch: {np.mean(times_arr):.6f} s")
        img_auroc = (
            roc_auc_score(img_true, img_pred_auroc)
            if len(np.unique(img_true)) > 1
            else np.nan
        )

        img_aupr = (
            average_precision_score(img_true, img_pred_auroc)
            if len(np.unique(img_true)) > 1
            else np.nan
        )

        px_true_arr = np.array(px_true_all, dtype=np.uint8)
        px_pred_arr_auroc = np.array(px_pred_all_auroc)
        px_pred_arr_normalized = np.array(px_pred_all_normalized)
        has_pos = (px_true_arr == 1).any()
        has_neg = (px_true_arr == 0).any()
        px_auroc = (
            roc_auc_score(px_true_arr, px_pred_arr_auroc)
            if (has_pos and has_neg)
            else np.nan
        )
        img_f1 = f1_score(img_true, img_pred_f1) if (thr_img is not None) else np.nan
        if thr_px is not None and has_pos:
            px_f1 = f1_score(
                px_true_arr.astype(int),
                (px_pred_arr_normalized >= thr_px).astype(int),
            )
        else:
            px_f1 = np.nan
        if len(anomalous_gt_masks) > 0:
            preds_np = np.stack(anomalous_anomaly_maps).astype(np.float32)  # [N,H,W]
            gts_np = np.stack(anomalous_gt_masks).astype(np.uint8)  # [N,H_W]
            preds_t = (
                torch.from_numpy(preds_np).unsqueeze(1).to(torch.float32).to(DEVICE)
            )  # [N,1,H,W]
            gts_t = (
                torch.from_numpy(gts_np).unsqueeze(1).to(torch.bool).to(DEVICE)
            )  # [N,1,H,W]

            fpr_cap = getattr(args, "pro_integration_limit", 0.3)
            tm_metric = TM_AUPRO(fpr_limit=fpr_cap).to(DEVICE)
            au_pro = tm_metric(preds_t, gts_t).item()
        else:
            logging.warning(
                f"No anomalous images found in test set for {category}. AUPRO is not computable."
            )
            au_pro = np.nan
        # ============================================================
        # Reviewer experiment: summarize residual-space separability
        # ============================================================
        if (
            len(residual_normal_list) > 0
            and len(residual_anomaly_list) > 0
            and len(nre_normal_list) > 0
            and len(nre_anomaly_list) > 0
        ):
            _rn = np.concatenate(residual_normal_list)
            _ra = np.concatenate(residual_anomaly_list)
            _nn = np.concatenate(nre_normal_list)
            _na = np.concatenate(nre_anomaly_list)

            # ============================================================
            # Reviewer experiment: save token-level residual distributions
            # ============================================================
            residual_save_dir = os.path.join(
                args.outdir, category, "residual_distribution"
            )
            os.makedirs(residual_save_dir, exist_ok=True)

            np.save(
                os.path.join(residual_save_dir, "normal_residual.npy"),
                _rn.astype(np.float32)
            )

            np.save(
                os.path.join(residual_save_dir, "anomaly_residual.npy"),
                _ra.astype(np.float32)
            )

            np.save(
                os.path.join(residual_save_dir, "normal_nre.npy"),
                _nn.astype(np.float32)
            )

            np.save(
                os.path.join(residual_save_dir, "anomaly_nre.npy"),
                _na.astype(np.float32)
            )

            logging.info(
                f"[Reviewer] Residual distributions saved to {residual_save_dir}"
            )
            # ============================================================

            _mu_n = float(np.mean(_rn))
            _mu_a = float(np.mean(_ra))
            _std_n = float(np.std(_rn))
            _std_a = float(np.std(_ra))
            _gap = _mu_a - _mu_n
            _fisher = (
                (_mu_a - _mu_n) ** 2
                / (_std_a ** 2 + _std_n ** 2 + 1e-12)
            )

            _res_labels = np.concatenate(
                [
                    np.zeros(_rn.shape[0], dtype=np.uint8),
                    np.ones(_ra.shape[0], dtype=np.uint8),
                ]
            )
            _res_scores = np.concatenate([_rn, _ra])
            _nre_scores = np.concatenate([_nn, _na])

            _res_auc = float(
                roc_auc_score(_res_labels, _res_scores)
            )
            _nre_auc = float(
                roc_auc_score(_res_labels, _nre_scores)
            )

            _method_name = (
                "MAD-RPCA"
                if getattr(args, "use_robust_pca", False)
                else "Standard PCA"
            )

            residual_results.append(
                {
                    "Category": category,
                    "Subspace Method": _method_name,
                    "Normal Residual Mean": _mu_n,
                    "Anomaly Residual Mean": _mu_a,
                    "Residual Gap": _gap,
                    "Fisher Ratio": _fisher,
                    "Residual AUROC": _res_auc,
                    "Normal NRE Mean": float(np.mean(_nn)),
                    "Anomaly NRE Mean": float(np.mean(_na)),
                    "NRE Gap": float(np.mean(_na) - np.mean(_nn)),
                    "NRE AUROC": _nre_auc,
                    "PCA Dim": int(pca_params["k"]),
                    "Normal Tokens": int(_rn.shape[0]),
                    "Anomaly Tokens": int(_ra.shape[0]),
                }
            )

            logging.info(
                f"[Residual] {category} | Method={_method_name} | "
                f"N={_mu_n:.6f} | A={_mu_a:.6f} | "
                f"Gap={_gap:.6f} | Fisher={_fisher:.6f} | "
                f"Res-AUROC={_res_auc:.6f} | "
                f"NRE-AUROC={_nre_auc:.6f} | "
                f"k={int(pca_params['k'])}"
            )

        if len(test_normal_tokens_list) > 0:
            aug_suffix = f"AUG_{args.aug_count}" if args.aug_count > 0 else "NO_AUG"
            # 同样加入 category 路径防止覆盖
            save_dir = os.path.join(args.outdir, category, "manifold_features")
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, f"{aug_suffix}_test_normal.npy"), np.concatenate(test_normal_tokens_list))
            if len(test_anomaly_tokens_list) > 0:
                np.save(os.path.join(save_dir, f"{aug_suffix}_test_anomaly.npy"),
                        np.concatenate(test_anomaly_tokens_list))
            print(f">> [Hook] Saved test features for {category}!")
        # 5. Log and store results
        logging.info(
            f"{category} Results | I-AUROC: {img_auroc:.4f} | I-AUPR: {img_aupr:.4f} | "
            f"P-AUROC: {px_auroc:.4f} | AU-PRO: {au_pro:.4f} | "
            f"I-F1: {img_f1:.4f} | P-F1: {px_f1:.4f}"
        )
        all_results.append(
            [category, img_auroc, img_aupr, px_auroc, au_pro, img_f1, px_f1]
        )
    df = pd.DataFrame(
        all_results,
        columns=[
            "Category",
            "Image AUROC",
            "Image AUPR",
            "Pixel AUROC",
            "AU-PRO",
            "Image F1",
            "Pixel F1",
        ],
    )
    if not df.empty and len(df) > 1:
        mean_values = df.mean(numeric_only=True)
        mean_row = pd.DataFrame(
            [["Average"] + mean_values.tolist()], columns=df.columns
        )
        df = pd.concat([df, mean_row], ignore_index=True)

    logging.info("\n--- Benchmark Final Results ---")
    logging.info("\n" + df.to_string(index=False, float_format="%.4f", na_rep="N/A"))

    results_path = os.path.join(args.outdir, "benchmark_results.csv")
    df.to_csv(results_path, index=False, float_format="%.4f")
    logging.info(f"\nResults saved to {results_path}")

    # Save reviewer residual-space metrics separately.
    if residual_results:
        residual_df = pd.DataFrame(residual_results)
        if len(residual_df) > 1:
            numeric_cols = residual_df.select_dtypes(
                include=[np.number]
            ).columns
            avg_values = residual_df[numeric_cols].mean()
            avg_row = {
                "Category": "Average",
                "Subspace Method": residual_df[
                    "Subspace Method"
                ].iloc[0],
            }
            for _col in numeric_cols:
                avg_row[_col] = avg_values[_col]
            residual_df = pd.concat(
                [residual_df, pd.DataFrame([avg_row])],
                ignore_index=True,
            )

        residual_path = os.path.join(
            args.outdir, "residual_space_metrics.csv"
        )
        residual_df.to_csv(
            residual_path, index=False, float_format="%.6f"
        )
        logging.info(
            f"\nResidual-space metrics saved to {residual_path}"
        )


if __name__ == "__main__":
    main()
