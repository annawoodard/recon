import time
from einops import rearrange
import torch
import torch.fft
import numpy as np
import pywt


def create_2d_gaussian_filter(nx, ny, mu_x, mu_y, sigma_x, sigma_y):
    """
    Create a 2D Gaussian filter.

    Args:
        nx (int): Size of the first dimension (width).
        ny (int): Size of the second dimension (height).
        mu_x (float): Mean of the Gaussian distribution along the x-axis.
        mu_y (float): Mean of the Gaussian distribution along the y-axis.
        sigma_x (float): Standard deviation of the Gaussian distribution along the x-axis.
        sigma_y (float): Standard deviation of the Gaussian distribution along the y-axis.

    Returns:
        torch.Tensor: 2D tensor containing the Gaussian filter.
    """
    start = time.time()

    x = torch.linspace(0, nx - 1, steps=nx)
    y = torch.linspace(0, ny - 1, steps=ny)
    x_grid, y_grid = torch.meshgrid(x, y, indexing="ij")

    gaussian_filter = torch.exp(
        -(
            (x_grid - mu_x) ** 2 / (2 * sigma_x**2)
            + (y_grid - mu_y) ** 2 / (2 * sigma_y**2)
        )
    )
    gaussian_filter /= gaussian_filter.sum()  # normalize

    print(f"created 2D Gaussian filter in {time.time() - start:.0e}s")

    return gaussian_filter


def create_1d_gaussian_filter(nx, ny, mu_x=86, sigma_base=42, sigma_offset_index=29):
    """
    Create Gaussian filters for each 'Y' dimension with dynamic standard deviation.

    Args:
        nx (int): Size of the first dimension (width).
        ny (int): Size of the second dimension (height).
        mu_x (int): Central x position for Gaussian mean.
        sigma_base (int): Base value for computing sigma.
        sigma_offset_index (int): Y index at which sigma is calculated based on distance.

    Returns:
        torch.Tensor: 2D tensor containing Gaussian filters for each 'Y'.
    """
    start = time.time()
    xvals = torch.linspace(1, nx, steps=nx).unsqueeze(1)  # shape [nx, 1]
    pe2_vals = torch.arange(ny)  # shape [ny]
    sigs = sigma_base - torch.abs(
        pe2_vals - sigma_offset_index
    )  # dynamic standard deviations
    gaussian_filters = torch.exp(-((xvals - mu_x) ** 2) / (2 * sigs**2))
    gaussian_filters /= gaussian_filters.sum(
        dim=0, keepdim=True
    )  # normalize each column
    print(f"created gaussian filters in {time.time() - start:.0e}s")
    return gaussian_filters


def apply_gaussian_filter(kspace, gaussian_filters):
    """
    Apply Gaussian filters to k-space data using broadcasting.

    Args:
        kspace (torch.Tensor): The k-space data tensor of shape [X, Y, Z, Num_Coils, Time].
        gaussian_filters (torch.Tensor): Gaussian filters for each 'Y', shape [X, Y].

    Returns:
        torch.Tensor: Filtered k-space data.
    """
    # Ensure Gaussian filters are broadcastable to the kspace dimensions
    gaussian_filters = (
        gaussian_filters.unsqueeze(2).unsqueeze(3).unsqueeze(4)
    )  # Shape [X, Y, 1, 1, 1]
    start = time.time()
    res = kspace * gaussian_filters  # broadcast the filter across Z, Num_Coils, Time
    print(f"applied filter in {time.time() - start:.1f}s")
    return res


def wavelet_kspace_filter(
    kspace,
    wavelet="db4",
    level=1,
    threshold_func="soft",
    sigma=None,
    threshold_factor=1.0,
):
    """
    Apply wavelet-based k-space filtering in PyTorch.
    Args:
        kspace (torch.Tensor): The input k-space data of shape (x, y, z, coils, time).
        wavelet (str): The wavelet family to use for the transform (default: 'db4').
        level (int): The number of wavelet decomposition levels (default: 1).
        threshold_func (str): The thresholding function to use ('soft' or 'hard', default: 'soft').
        sigma (float or None): The noise standard deviation. If None, it will be estimated.
    Returns:
        torch.Tensor: The filtered k-space data.
    """

    def soft_threshold(coeffs, threshold):
        abs_coeffs = torch.abs(coeffs)
        return torch.sign(coeffs) * torch.max(
            abs_coeffs - threshold, torch.zeros_like(abs_coeffs)
        )

    def hard_threshold(coeffs, threshold):
        return coeffs * (torch.abs(coeffs) > threshold)

    # Wavelet transform along all spatial dimensions using PyTorch
    coeffs = pywt.wavedecn(
        kspace.numpy(), wavelet=wavelet, level=level, mode="periodic"
    )
    arr, coeff_slices = coeffs[0], coeffs[1:]

    # Flatten all coefficient arrays contained within the dictionaries
    flat_coeffs = torch.cat([torch.tensor(c["detail"].flatten()) for c in coeff_slices])

    # Estimate noise standard deviation if not provided
    if sigma is None:
        sigma = 1.4826 * torch.median(
            torch.abs(flat_coeffs - torch.median(flat_coeffs))
        )

    # Determine the thresholding function
    if threshold_func == "soft":
        thresholder = soft_threshold
    elif threshold_func == "hard":
        thresholder = hard_threshold
    else:
        raise ValueError("Invalid thresholding function. Choose 'soft' or 'hard'.")

    # Apply thresholding
    threshold = (
        threshold_factor
        * sigma
        * torch.sqrt(2 * torch.log(torch.tensor(arr.size).prod()))
    )
    denoised_arr = thresholder(torch.tensor(arr), threshold)

    # Reconstruct the signal using the thresholded coefficients
    denoised_coeffs = [denoised_arr.numpy()] + coeff_slices
    filtered_kspace = pywt.waverecn(denoised_coeffs, wavelet=wavelet, mode="periodic")

    return torch.tensor(filtered_kspace, device=kspace.device)


def estimate_noise_level(kspace, patch_size):
    """
    Estimate the local noise level in the k-space using a sliding window approach.
    """
    padding = patch_size // 2
    padded_kspace = torch.zeros(
        (
            kspace.shape[0] + 2 * padding,
            kspace.shape[1] + 2 * padding,
            kspace.shape[2],
            kspace.shape[3],
            kspace.shape[4],
        ),
        dtype=kspace.dtype,
        device=kspace.device,
    )
    padded_kspace[padding:-padding, padding:-padding, :, :, :] = kspace
    padded_kspace[:padding, padding:-padding, :, :, :] = kspace[0, :, :, :, :]
    padded_kspace[-padding:, padding:-padding, :, :, :] = kspace[-1, :, :, :, :]
    padded_kspace[:, :padding, :, :, :] = padded_kspace[
        :, padding : padding + 1, :, :, :
    ]
    padded_kspace[:, -padding:, :, :, :] = padded_kspace[
        :, -padding - 1 : -padding, :, :, :
    ]

    noise_level = torch.zeros_like(kspace)

    for i in range(padding, padding + kspace.shape[0]):
        for j in range(padding, padding + kspace.shape[1]):
            patch = padded_kspace[
                i - padding : i + padding + 1, j - padding : j + padding + 1, :, :, :
            ]
            noise_level[i - padding, j - padding, :, :, :] = torch.std(patch)

    return noise_level


def adaptive_kspace_filter(kspace, patch_size, threshold_factor):
    """
    Apply adaptive k-space filtering based on local noise level estimation and thresholding.
    """
    noise_level = estimate_noise_level(torch.abs(kspace), patch_size)
    threshold = threshold_factor * noise_level

    filtered_kspace = kspace.clone()
    filtered_kspace[torch.abs(kspace) < threshold] = 0

    return filtered_kspace


def svd_kspace_filter(kspace, rank=None, threshold=None):
    """
    Apply SVD-based k-space filtering.

    Args:
        kspace (torch.Tensor): The input k-space data of shape (x, y, z, coils, time).
        rank (int or None): The rank of the truncated SVD. If None, no truncation is performed.
        threshold (float or None): The threshold value for singular value thresholding. If None, no thresholding is performed.

    Returns:
        torch.Tensor: The filtered k-space data.
    """
    # Rearrange the k-space data to a 2D matrix using einops
    kspace_matrix = rearrange(kspace, "x y z c t -> (c t) (x y z)")

    # Compute the SVD of the real and imaginary parts separately
    U_real, S_real, Vh_real = torch.linalg.svd(kspace_matrix.real, full_matrices=False)
    U_imag, S_imag, Vh_imag = torch.linalg.svd(kspace_matrix.imag, full_matrices=False)

    # Truncate the SVD if rank is specified
    if rank is not None:
        U_real = U_real[:, :rank]
        S_real = S_real[:rank]
        Vh_real = Vh_real[:rank, :]
        U_imag = U_imag[:, :rank]
        S_imag = S_imag[:rank]
        Vh_imag = Vh_imag[:rank, :]

    # Apply singular value thresholding if threshold is specified
    if threshold is not None:
        S_real = torch.max(S_real - threshold, torch.zeros_like(S_real))
        S_imag = torch.max(S_imag - threshold, torch.zeros_like(S_imag))

    # Reconstruct the filtered k-space matrix for real and imaginary parts separately
    filtered_kspace_real = torch.matmul(
        U_real, torch.matmul(torch.diag(S_real), Vh_real)
    )
    filtered_kspace_imag = torch.matmul(
        U_imag, torch.matmul(torch.diag(S_imag), Vh_imag)
    )

    # Rearrange the real and imaginary parts back to the original shape using einops
    filtered_kspace_real = rearrange(
        filtered_kspace_real,
        "(c t) (x y z) -> x y z c t",
        x=kspace.shape[0],
        y=kspace.shape[1],
        z=kspace.shape[2],
        c=kspace.shape[3],
        t=kspace.shape[4],
    )
    filtered_kspace_imag = rearrange(
        filtered_kspace_imag,
        "(c t) (x y z) -> x y z c t",
        x=kspace.shape[0],
        y=kspace.shape[1],
        z=kspace.shape[2],
        c=kspace.shape[3],
        t=kspace.shape[4],
    )

    # Combine the real and imaginary parts to form the filtered complex k-space data
    filtered_kspace = torch.complex(filtered_kspace_real, filtered_kspace_imag)

    return filtered_kspace
