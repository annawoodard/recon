import time
import torch
import torch.fft


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
    gaussian_filter /= gaussian_filter.sum()  # Normalize the filter

    print(f"Created 2D Gaussian filter in {time.time() - start:.0e}s")

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
