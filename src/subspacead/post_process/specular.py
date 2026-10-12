import torch
import kornia as K
import numpy as np

EPS = 1e-6


def _get_brightness_cue(Y: torch.Tensor) -> torch.Tensor:
    """Calculates the brightness cue (sY)."""
    kY, tY = 15.0, 0.65
    return torch.sigmoid(kY * (Y - tY))


def _get_desaturation_cue(S: torch.Tensor) -> torch.Tensor:
    """Calculates the desaturation cue (sS)."""
    kS, tS = 10.0, 0.3
    return torch.sigmoid(kS * (tS - S))


def _get_curvature_cue(Y: torch.Tensor, B: int) -> torch.Tensor:
    """Calculates the curvature cue (sK) using Laplacian of Gaussian."""
    Y_blur = K.filters.gaussian_blur2d(Y, (3, 3), (1.0, 1.0))
    lap = K.filters.laplacian(Y_blur, kernel_size=3)  # [B,1,H,W]
    tk = torch.quantile(lap.view(B, -1), q=0.95, dim=1).view(B, 1, 1, 1) + EPS
    return torch.sigmoid(4.0 * (lap - tk) / tk)


def specular_mask_torch(img_rgb: torch.Tensor, tau: float = 0.6):
    """
    Generates a specular mask from an sRGB image tensor.

    Args:
        img_rgb: float tensor in [0,1], shape [B,3,H,W], sRGB
        tau: Binarization threshold for the mask.

    Returns:
        bin_mask (torch.Tensor): [B,1,H,W] bool mask (True where specular)
        soft_spec (torch.Tensor): [B,1,H,W] float mask [0,1]
        conf (torch.Tensor): [B,1,H,W] float confidence [0,1] (1.0 - soft_spec)
    """
    B, C, H, W = img_rgb.shape

    # Linearize sRGB -> RGB
    I_lin = torch.clamp(img_rgb, EPS, 1.0) ** 2.2
    R, G, Bc = I_lin[:, 0:1], I_lin[:, 1:2], I_lin[:, 2:3]
    # Luminance (Y)
    Y = 0.2126 * R + 0.7152 * G + 0.0722 * Bc
    # Saturation (S)
    S = K.color.rgb_to_hsv(img_rgb)[:, 1:2]

    clip_flag = (img_rgb.max(dim=1, keepdim=True).values > 0.985).float()
    sY = _get_brightness_cue(Y)
    sS = _get_desaturation_cue(S)
    sK = _get_curvature_cue(Y, B)

    w1, w2, w3, w4 = 0.5, 0.3, 0.2, 0.3
    Sspec = torch.clamp(w1 * sY + w2 * sS + w3 * sK + w4 * clip_flag, 0.0, 1.0)

    bin_mask = Sspec > tau
    conf = 1.0 - Sspec

    return bin_mask, Sspec, conf


def _prepare_tensor(
    tensor: torch.Tensor | np.ndarray, device: torch.device
) -> (torch.Tensor, tuple, torch.device):
    """
    Converts input (numpy or tensor) to a 4D tensor on the correct device.
    Returns the 4D tensor, original shape, and the device.
    """
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    elif not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"Input must be a torch.Tensor or np.ndarray. Got: {type(tensor)}"
        )

    tensor = tensor.to(device)
    original_shape = tensor.shape

    # Reshape to 4D [B, 1, H, W] for kornia filters
    if tensor.dim() == 2:  # [H, W]
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 3:  # [B, H, W]
        tensor = tensor.unsqueeze(1)

    if tensor.dim() != 4:
        raise ValueError(f"Could not convert input with shape {original_shape} to 4D.")

    return tensor, original_shape


def filter_specular_anomalies(
    anomaly_map: torch.Tensor | np.ndarray,
    conf_map: torch.Tensor | np.ndarray,
    blur_sigma: float = 5.0,
) -> torch.Tensor:
    """
    Filters specular FPs by comparing a pixel's anomaly score to its
    non-specular neighborhood.
    """
    ksize = int(blur_sigma * 4 + 0.5) * 2 + 1
    blur_kernel = (ksize, ksize), (blur_sigma, blur_sigma)

    # Determine target device from inputs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(conf_map, torch.Tensor) and conf_map.is_cuda:
        device = conf_map.device
    elif isinstance(anomaly_map, torch.Tensor) and anomaly_map.is_cuda:
        device = anomaly_map.device

    conf_map_4d, _ = _prepare_tensor(conf_map, device)
    anomaly_map_4d, original_shape = _prepare_tensor(anomaly_map, device)

    # Get the anomaly map for non-specular regions
    anomaly_map_non_spec = anomaly_map_4d * conf_map_4d

    # Get the average non-specular anomaly score in the neighborhood
    sum_weighted_anomalies = K.filters.gaussian_blur2d(
        anomaly_map_non_spec, *blur_kernel
    )
    sum_weights = K.filters.gaussian_blur2d(conf_map_4d, *blur_kernel)
    anomaly_map_non_spec_avg = sum_weighted_anomalies / (sum_weights + EPS)

    # Compute the "context score"
    context_score = (anomaly_map_non_spec_avg / (anomaly_map_4d + EPS)).clamp(0.0, 1.0)

    # Linearly interpolate the suppression multiplier
    suppression_multiplier = torch.lerp(
        conf_map_4d,
        torch.tensor(1.0, device=device),
        context_score,
    )

    filtered_map = (anomaly_map_4d * suppression_multiplier).clone().detach()

    if len(original_shape) == 2:
        return filtered_map.squeeze(0).squeeze(0)  # [1, 1, H, W] -> [H, W]
    elif len(original_shape) == 3:
        return filtered_map.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
    else:
        return filtered_map  # Already [B, 1, H, W]
