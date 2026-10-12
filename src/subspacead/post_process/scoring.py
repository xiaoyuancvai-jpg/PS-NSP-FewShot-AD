import cv2
import numpy as np
import logging
from subspacead.utils.common import topk_mean


def aggregate_image_score(anomaly_map: np.ndarray, method: str) -> float:
    """
    Aggregates a pixel-level anomaly map into a single image-level score.
    """
    if method == "max":
        return float(np.max(anomaly_map))
    elif method == "p99":
        return float(np.percentile(anomaly_map, 99))
    elif method == "mtop5":
        return float(np.mean(np.sort(anomaly_map.flatten())[-5:]))
    elif method == "mtop1p":
        return topk_mean(anomaly_map, frac=0.01)
    elif method == "mean":
        return float(np.mean(anomaly_map))
    else:
        logging.warning(
            f"Unknown image score aggregation '{method}'. Defaulting to 'mean'."
        )
        return float(np.mean(anomaly_map))


def _kernel_self_dot(X: np.ndarray, kpca) -> np.ndarray:
    """Compute k(x,x) for several sklearn KPCA kernels."""
    if kpca.kernel in ("rbf", "cosine"):
        return np.ones(X.shape[0])
    elif kpca.kernel == "linear":
        return np.sum(X**2, axis=1) + kpca.coef0
    elif kpca.kernel == "poly":
        gamma = kpca.gamma if kpca.gamma is not None else 1.0 / X.shape[1]
        return (gamma * np.sum(X**2, axis=1) + kpca.coef0) ** kpca.degree
    elif kpca.kernel == "sigmoid":
        gamma = kpca.gamma if kpca.gamma is not None else 1.0 / X.shape[1]
        return np.tanh(gamma * np.sum(X**2, axis=1) + kpca.coef0)
    else:
        logging.warning(
            f"Cannot compute k(x,x) for kernel '{kpca.kernel}'. Reconstruction error will be approximate."
        )
        return np.zeros(X.shape[0], dtype=X.dtype)


def _row_l2(X: np.ndarray, eps: float) -> np.ndarray:
    """Normalize rows to unit length."""
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + eps)


def pca_reconstruct(X: np.ndarray, pca: dict, drop_k: int = 0) -> np.ndarray:
    """Reconstruct X in original space using unscaled components."""
    mu = np.asarray(pca["mu"], dtype=X.dtype)
    C = np.asarray(pca["components"][:, : pca["k"]], dtype=X.dtype)
    X0 = X - mu
    Z = X0 @ C
    if drop_k > 0:
        if drop_k >= Z.shape[1]:
            Z[:] = 0.0
        else:
            Z[:, :drop_k] = 0.0
    X_recon = (Z @ C.T) + mu
    return X_recon


def _calculate_kpca_scores(X: np.ndarray, pca: dict, drop_k: int = 0):
    """Calculates anomaly scores for Kernel PCA."""
    scaler = pca["scaler"]
    kpca = pca["kpca"]
    X_scaled = scaler.transform(X)

    X_proj = kpca.transform(X_scaled)
    k_x_x = _kernel_self_dot(X_scaled, kpca)

    if drop_k > 0:
        if drop_k >= X_proj.shape[1]:
            X_proj = np.zeros_like(X_proj)
        else:
            X_proj = X_proj[:, drop_k:]

        proj_norm_sq = np.sum(X_proj**2, axis=1)
        score = k_x_x - proj_norm_sq
        return np.maximum(0.0, score)


def _calculate_pca_scores(X: np.ndarray, pca: dict, method: str, drop_k: int = 0):
    """Calculates anomaly scores for standard PCA."""
    if drop_k < 0:
        raise ValueError("drop_k must be non-negative.")
    if drop_k >= pca["k"]:
        logging.warning(f"drop_k ({drop_k}) is >= num components ({pca['k']}).")
        if method in ("mahalanobis", "euclidean"):
            return np.zeros(X.shape[0], dtype=X.dtype)

    if method == "reconstruction":
        X_recon = pca_reconstruct(X, pca, drop_k=drop_k)
        return np.sum((X - X_recon) ** 2, axis=1)

    elif method == "mahalanobis":
        mu = np.asarray(pca["mu"], dtype=X.dtype)
        C = np.asarray(pca["components"][:, : pca["k"]], dtype=X.dtype)
        Z = (X - mu) @ C  # [N, k]

        if drop_k >= pca["k"]:
            return np.zeros(X.shape[0], dtype=X.dtype)

        Z_abnormal = Z[:, drop_k:]
        eigvals_abnormal = np.asarray(pca["eigvals"][drop_k:], dtype=X.dtype)
        cov_inv = np.diag(1.0 / (eigvals_abnormal + pca["eps"]))
        return np.einsum("ij,jk,ik->i", Z_abnormal, cov_inv, Z_abnormal)

    elif method == "euclidean":
        mu = np.asarray(pca["mu"], dtype=X.dtype)
        C = np.asarray(pca["components"][:, : pca["k"]], dtype=X.dtype)
        Z = (X - mu) @ C  # [N, k]

        if drop_k >= pca["k"]:
            return np.zeros(X.shape[0], dtype=X.dtype)

        Z_abnormal = Z[:, drop_k:]
        return np.sum(Z_abnormal**2, axis=1)

    elif method == "cosine":
        X_recon = pca_reconstruct(X, pca, drop_k=drop_k)
        X_norm = _row_l2(X, pca["eps"])
        X_recon_norm = _row_l2(X_recon, pca["eps"])
        sim = np.einsum("ij,ij->i", X_norm, X_recon_norm)
        sim = np.clip(sim, -1.0, 1.0)
        return 1.0 - sim

    else:
        raise ValueError(f"Unknown scoring method '{method}'.")


def calculate_anomaly_scores(X: np.ndarray, pca: dict, method: str, drop_k: int = 0):
    """
    Calculates anomaly scores using PCA or KernelPCA.
    Acts as a router to the appropriate scoring function.
    """
    if "kpca" in pca:
        if method != "reconstruction":
            logging.warning(
                "Kernel PCA only supports 'reconstruction' scoring. Using 'reconstruction'."
            )
        return _calculate_kpca_scores(X, pca, drop_k)
    else:
        return _calculate_pca_scores(X, pca, method, drop_k)


def post_process_map(
    anomaly_map: np.ndarray,
    res,
    input_img: np.ndarray = None,
    blur: bool = True,
    close_holes: bool = False,
    close_k_size: int = 5,
):
    """Resize and blur the anomaly map (Canny Edge Penalty Completely REMOVED)."""
    if anomaly_map.dtype != np.float32:
        anomaly_map = anomaly_map.astype(np.float32)

    dsize = (res, res) if isinstance(res, int) else (res[1], res[0])
    map_resized = cv2.resize(anomaly_map, dsize, interpolation=cv2.INTER_LINEAR)

    # 🚨 Canny 边缘惩罚已被彻底删除！现在边缘的真实缺陷将得到 100% 的保留！

    if close_holes:
        if close_k_size % 2 == 0:
            close_k_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k_size, close_k_size))
        map_resized = cv2.morphologyEx(map_resized, cv2.MORPH_CLOSE, kernel)

    if blur:
        sigma = 4.0
        k_size = 3
        return cv2.GaussianBlur(map_resized, (k_size, k_size), sigma)
    else:
        return map_resized