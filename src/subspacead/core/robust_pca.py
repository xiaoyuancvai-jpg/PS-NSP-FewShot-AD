import torch
import numpy as np


def augment_aware_weighted_pca(X, pca_ev=0.99, alpha=2.0, w_min=0.01, tau=2.0, **kwargs):
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)

    # 取消全局强制转 Float64，让海量数据的矩阵 X 保持 Float32 省显存
    # X = X.to(dtype=torch.float64)  <- 把这行删掉或注释掉
    X = X.to(device=X.device, dtype=torch.float32)
    N, C = X.shape

    # =====================================================================
    # [阶段一] 第一轮基础 PCA，拟合粗糙子空间
    # =====================================================================
    mean_1 = torch.mean(X, dim=0)
    X_centered_1 = X - mean_1

    # 💥 关键修改：只在矩阵乘法的这一步，临时升维到 Float64 防止溢出
    cov_1 = torch.matmul(X_centered_1.to(torch.float64).T, X_centered_1.to(torch.float64)) / (N - 1)

    eigenvalues_1, eigenvectors_1 = torch.linalg.eigh(cov_1)
    # 计算完特征向量后，立刻转回 Float32，释放压力
    eigenvectors_1 = eigenvectors_1.to(torch.float32)
    eigenvalues_1 = eigenvalues_1.to(torch.float32)

    idx_1 = torch.argsort(eigenvalues_1, descending=True)
    eigenvectors_1 = eigenvectors_1[:, idx_1]

    cum_ev = torch.cumsum(eigenvalues_1[idx_1], dim=0) / torch.sum(eigenvalues_1)
    k_1 = torch.where(cum_ev >= pca_ev)[0][0].item() + 1
    V_1 = eigenvectors_1[:, :k_1]

    # 计算特征空间中的重构残差 (Residuals)
    X_rec_1 = torch.matmul(torch.matmul(X_centered_1, V_1), V_1.T)
    residuals = torch.norm(X_centered_1 - X_rec_1, p=2, dim=1)

    # =====================================================================
    # [阶段二] 鲁棒自适应加权 (核心：惩罚高残差的增强伪影)
    # =====================================================================
    med_res = torch.median(residuals)
    mad_res = torch.median(torch.abs(residuals - med_res))
    scale_res = 1.4826 * mad_res + 1e-6
    z_res = torch.clamp((residuals - med_res) / scale_res, min=0.0)

    # 💥 指数衰减：残差越大的伪影权重越接近 0；正常流形权重保留为 1
    w = torch.exp(- (z_res ** 2) / (2 * (tau ** 2)))
    w = torch.clamp(w, min=w_min, max=1.0)

    # =====================================================================
    # [阶段三] 提取最终的纯净子空间 (Weighted PCA)
    # =====================================================================
    # =====================================================================
    # [阶段三] 提取最终的纯净子空间 (Weighted PCA)
    # =====================================================================
    w_sum = torch.sum(w)
    w_mean = torch.sum(X * w[:, None], dim=0) / w_sum
    X_centered_2 = X - w_mean

    Xw = X_centered_2 * torch.sqrt(w)[:, None]

    # 💥 关键修改：Phase 3 同样启用临时 Float64 护航，彻底杜绝 NaN 溢出
    cov_2 = torch.matmul(Xw.to(torch.float64).T, Xw.to(torch.float64)) / w_sum.to(torch.float64)

    eigenvalues_2, eigenvectors_2 = torch.linalg.eigh(cov_2)
    # 计算完毕立刻转回 Float32
    eigenvalues_2 = eigenvalues_2.to(torch.float32)
    eigenvectors_2 = eigenvectors_2.to(torch.float32)
    idx_2 = torch.argsort(eigenvalues_2, descending=True)
    eigenvalues_2 = eigenvalues_2[idx_2]
    eigenvectors_2 = eigenvectors_2[:, idx_2]

    cum_ev_2 = torch.cumsum(eigenvalues_2, dim=0) / torch.sum(eigenvalues_2)
    k_2 = torch.where(cum_ev_2 >= pca_ev)[0][0].item() + 1
    V_final = eigenvectors_2[:, :k_2]

    # ✅ 完美返回 3 个参数
    return V_final.cpu().numpy(), w_mean.cpu().numpy(), eigenvalues_2[:k_2].cpu().numpy()