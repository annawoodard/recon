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
