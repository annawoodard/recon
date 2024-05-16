import time
import torch
import torch.fft


def fftshift3d(x):
    """Shift the zero-frequency component to the center of the spectrum."""
    return torch.fft.fftshift(x, dim=(0, 1, 2))


def ifftshift3d(x):
    """Undo the shift of the zero-frequency component from the center of the spectrum."""
    return torch.fft.ifftshift(x, dim=(0, 1, 2))


def ifft3d(x):
    """Apply three-dimensional inverse FFT with necessary shifts."""
    start = time.time()
    x = ifftshift3d(x)  # Undo the shift before applying ifft
    x = torch.fft.ifftn(
        x, dim=(0, 1, 2)
    )  # Apply inverse FFT across all three spatial dimensions
    x = fftshift3d(x)  # Shift back after ifft
    print(f"completed ifft3d in {time.time() - start:.2f} seconds")
    return x


def subtract_kspace(kspace, n_subtract):
    """
    Perform k-space subtraction for all time points.
    Args:
        kspace (torch.Tensor): The input k-space data of shape (x, y, z, coils, time).
        n_subtract (int): The number of time points to subtract.
    Returns:
        torch.Tensor: The subtracted k-space data of shape (x, y, z, coils, time).
    """
    if n_subtract is None or n_subtract <= 0:
        return kspace

    nt = kspace.shape[-1]
    subtracted_kspace = torch.zeros_like(kspace)

    for time_idx in range(nt):
        if time_idx - n_subtract >= 0:
            subtracted_kspace[..., time_idx] = (
                kspace[..., time_idx] - kspace[..., time_idx - n_subtract]
            )
        else:
            subtracted_kspace[..., time_idx] = kspace[..., time_idx]

    return subtracted_kspace
