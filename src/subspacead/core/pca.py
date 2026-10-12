import logging
import torch
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler


class KernelPCAModel:
    """Wraps sklearn.decomposition.KernelPCA for feature collection."""

    def __init__(self, k=None, kernel="rbf", gamma=None, eps=1e-6):
        self.k = k
        self.kernel = kernel
        self.gamma = gamma
        self.eps = eps
        self.scaler = None
        self.kpca = None
        self.pca_params = {}

    def fit(self, features: np.ndarray):
        logging.info("Starting Kernel PCA fit...")
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)

        self.kpca = KernelPCA(
            n_components=self.k,
            kernel=self.kernel,
            gamma=self.gamma,
            copy_X=False,
        )

        logging.info(f"Fitting KernelPCA with kernel='{self.kernel}'...")
        self.kpca.fit(features_scaled)
        self.pca_params = {
            "scaler": self.scaler,
            "kpca": self.kpca,
            "k": self.k,
            "eps": self.eps,
        }

        logging.info("Kernel PCA fit complete.")
        return self.pca_params


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"PCAModel will use device: {device}")


class PCAModel:
    """
    Memory-efficient PCA using a two-pass streaming algorithm on GPU.
    Based on https://github.com/dnhkng/PCAonGPU
    """

    def __init__(self, k=None, ev=None, whiten=False, eps=1e-6):
        self.k = k
        self.ev_ratio = ev
        self.whiten = whiten
        self.eps = eps
        self.mu_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.eigvals_ = None
        self.pca_params = {}
        self.device = device
        self.dtype = torch.float64

    def _compute_mean(self, feature_generator, feature_dim, total_tokens, num_batches):
        """Pass 1: Compute the mean of all features."""
        logging.info("Starting PCA Pass 1/2 (Mean)...")
        self.mu_ = torch.zeros(feature_dim, dtype=self.dtype, device=self.device)
        for batch in tqdm(
            feature_generator(), total=num_batches, desc="PCA Pass 1/2 (Mean)"
        ):
            batch_gpu = torch.from_numpy(batch).to(self.device, dtype=self.dtype)
            self.mu_ += torch.sum(batch_gpu, axis=0)
        self.mu_ /= total_tokens

    def _compute_covariance(
        self, feature_generator, feature_dim, total_tokens, num_batches
    ):
        """Pass 2: Compute the covariance matrix."""
        logging.info("Starting PCA Pass 2/2 (Covariance)...")
        cov_matrix = torch.zeros(
            (feature_dim, feature_dim), dtype=self.dtype, device=self.device
        )
        for batch in tqdm(
            feature_generator(), total=num_batches, desc="PCA Pass 2/2 (Cov)"
        ):
            batch_gpu = torch.from_numpy(batch).to(self.device, dtype=self.dtype)
            batch_centered = batch_gpu - self.mu_
            cov_matrix += torch.matmul(batch_centered.T, batch_centered)
        cov_matrix /= total_tokens - 1
        return cov_matrix

    def _compute_eigendecomposition(self, cov_matrix):
        """Perform eigendecomposition on the covariance matrix."""
        logging.info("Performing eigendecomposition on GPU...")
        evals, evecs = torch.linalg.eigh(cov_matrix)
        sorted_indices = torch.argsort(evals, descending=True)
        self.explained_variance_ = evals[sorted_indices]
        return evecs[:, sorted_indices]

    def _select_k_components(self, evecs):
        """Select the number of components (k) based on 'ev_ratio' or 'k'."""
        if self.ev_ratio is not None and self.k is None:
            cumulative_variance = torch.cumsum(
                self.explained_variance_, dim=0
            ) / torch.sum(self.explained_variance_)
            self.k = (
                torch.searchsorted(
                    cumulative_variance,
                    torch.tensor([self.ev_ratio], dtype=self.dtype, device=self.device),
                ).item()
                + 1
            )
            logging.info(
                f"PCA: selected k={self.k} components to explain {self.ev_ratio * 100:.2f}% of variance."
            )

        if self.k is None:
            self.k = evecs.shape[1]
        else:
            self.k = min(self.k, evecs.shape[1])

        self.components_ = evecs[:, : self.k]
        self.eigvals_ = self.explained_variance_[: self.k]

    def _build_pca_params(self):
        """Copies GPU tensors to a CPU numpy dictionary for pipeline use."""
        self.pca_params = {
            "mu": self.mu_.cpu().numpy().astype(np.float64),
            "components": self.components_.cpu().numpy().astype(np.float64),
            "eigvals": self.eigvals_.cpu().numpy().astype(np.float64),
            "sqrt_eig": np.sqrt(
                self.eigvals_.cpu().numpy().astype(np.float64) + self.eps
            ),
            "k": self.k,
            "whiten": self.whiten,
            "eps": self.eps,
            "cov_Z_inv": np.diag(
                1.0 / (self.eigvals_.cpu().numpy().astype(np.float64) + self.eps)
            ),
        }
        return self.pca_params

    def fit(
        self, feature_generator, feature_dim: int, total_tokens: int, num_batches: int
    ):
        """
        Orchestrates the two-pass streaming PCA fit.
        """
        logging.info(f"Starting PCA fit on {self.device}...")

        self._compute_mean(feature_generator, feature_dim, total_tokens, num_batches)

        cov_matrix = self._compute_covariance(
            feature_generator, feature_dim, total_tokens, num_batches
        )

        evecs = self._compute_eigendecomposition(cov_matrix)

        self._select_k_components(evecs)

        return self._build_pca_params()
