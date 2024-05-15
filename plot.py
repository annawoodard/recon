import matplotlib.pyplot as plt
import cv2
import torch
import torch.fft
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.patches as patches


def normalize_image(image, window=None):
    """
    Normalize the image based on the maximum intensity within a specified window.

    Args:
        image (torch.Tensor): The input image tensor.
        window (int, optional): The height of the window to use for finding the maximum intensity.
    Returns:
        torch.Tensor: The normalized image tensor.
    """
    if window is not None:
        # Ensure window does not exceed image dimensions
        window = min(window, image.shape[0])
        windowed_image = image[:window, ...]
    else:
        windowed_image = image

    max_val = torch.max(windowed_image)
    return image / max_val if max_val > 0 else image


def get_windowed_bounds(image, lower_p, upper_p, window_height):
    """
    Calculate the lower and upper bounds of the image within the specified window height.

    Args:
        image (torch.Tensor): The image tensor.
        lower_p (float or None): Lower percentile to calculate (0-100), or None to use the minimum value.
        upper_p (float or None): Upper percentile to calculate (0-100), or None to use the maximum value.
        window_height (int): The height of the window to consider for percentile calculation.
    Returns:
        float, float: The lower and upper percentile values within the window, or min/max values if None.
    """
    if window_height is not None and window_height < image.shape[0]:
        windowed_image = image[:window_height, ...]
    else:
        windowed_image = image

    if lower_p is not None:
        lower_quantile = lower_p / 100.0
        vmin = windowed_image.quantile(lower_quantile).item()
    else:
        vmin = windowed_image.min().item()

    if upper_p is not None:
        upper_quantile = upper_p / 100.0
        vmax = windowed_image.quantile(upper_quantile).item()
    else:
        vmax = windowed_image.max().item()

    return vmin, vmax


def plot_coil_images(
    img_space,
    time_idx,
    tr,
    n_subtract=None,
    window=None,
    lower_percentile=None,
    upper_percentile=None,
    tag=None,
):
    """
    Plot coil images from k-space MRI data.
    Args:
        img_space (torch.Tensor): The input image space data of shape (x, y, z, coils, time).
        time_idx (int): The time index to visualize.
        tr (float): The repetition time (TR) in seconds.
        n_subtract (int, optional): The number of time points to subtract for the subtraction image.
        window (int): The point to cut images to get rid of thorax/heart signal.
    """
    nx, ny, nz, num_coils, nt = img_space.shape
    magnitude = torch.abs(img_space)

    if n_subtract is not None and (time_idx - n_subtract >= 0):
        curr_images = magnitude[:, :, :, :, time_idx]
        past_images = magnitude[:, :, :, :, time_idx - n_subtract]
        subtracted_images = curr_images - past_images
    else:
        subtracted_images = magnitude[:, :, :, :, time_idx]

    mip_magnitude = torch.max(subtracted_images, dim=2).values
    fig, axs = plt.subplots(1, num_coils, figsize=(25, 10))
    for i in range(num_coils):
        vmin, vmax = get_windowed_bounds(
            mip_magnitude[:, :, i], lower_percentile, upper_percentile, window
        )

        axs[i].imshow(
            mip_magnitude[:, :, i].numpy(),
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            interpolation="none",
        )
        axs[i].axis("off")
        axs[i].set_title(f"{i+1}", fontsize=12)
        if window is not None:
            # red dotted rectangle around the windowed area
            axs[i].add_patch(
                patches.Rectangle(
                    (0, 0),
                    mip_magnitude.shape[1],
                    window,
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                    linestyle="--",
                )
            )

    plt.suptitle(
        ("subtraction MIP " if n_subtract is not None else "MIP ")
        + f"at {time_idx * tr:.1f}s (time point {time_idx}, TR {tr:.2f}{f', n_subtract {n_subtract}' if n_subtract is not None else ''})"
        + (f"\n{tag}" if tag is not None else ""),
        fontsize=20,
    )
    if tag is not None:
        tag = tag.replace(" ", "_")
    plt.savefig(
        f"plots/coil_images_tp_{time_idx}{f'_{tag}' if tag is not None else ''}.png"
    )
    plt.tight_layout()


def plot_recon_comparison(
    phillips_recon,
    precon_recon,
    time_idx,
    tr,
    n_subtract=None,
    window=None,
    lower_percentile=None,
    upper_percentile=None,
):
    magnitude_phillips = torch.abs(phillips_recon[:, :, :, time_idx])
    magnitude_precon = torch.abs(precon_recon[:, :, :, time_idx])

    if n_subtract is not None and (time_idx - n_subtract >= 0):
        past_phillips = torch.abs(phillips_recon[:, :, :, time_idx - n_subtract])
        past_precon = torch.abs(precon_recon[:, :, :, time_idx - n_subtract])

        magnitude_phillips -= past_phillips
        magnitude_precon -= past_precon

    if torch.isnan(magnitude_precon).any():
        print("Nan values detected in precon")
        magnitude_precon = torch.nan_to_num(magnitude_precon)

    mip_phillips = torch.max(magnitude_phillips, dim=2).values
    mip_precon = torch.max(magnitude_precon, dim=2).values

    vmin_phillips, vmax_phillips = get_windowed_bounds(
        mip_phillips, lower_percentile, upper_percentile, window
    )
    vmin_precon, vmax_precon = get_windowed_bounds(
        mip_precon, lower_percentile, upper_percentile, window
    )

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(
        mip_phillips.numpy(),
        cmap="gray",
        vmin=vmin_phillips,
        vmax=vmax_phillips,
        interpolation="none",
    )
    axs[0].axis("off")
    axs[0].set_title("phillips recon")
    axs[1].imshow(
        mip_precon.numpy(),
        cmap="gray",
        vmin=vmin_precon,
        vmax=vmax_precon,
        interpolation="none",
    )
    axs[1].axis("off")
    axs[1].set_title("precon recon")

    # Add a red dotted rectangle to indicate the window area
    if window is not None:
        for ax in axs:
            ax.add_patch(
                patches.Rectangle(
                    (0, 0),
                    mip_phillips.shape[1],
                    window,
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                    linestyle="--",
                )
            )

    plt.tight_layout()


def plot_recon(
    recon,
    time_idx,
    tr,
    n_subtract=None,
    window=None,
    lower_percentile=None,
    upper_percentile=None,
):
    """
    Plot the Philips reconstruction for a specific time index, performing subtraction before MIP if specified.
    Args:
        recon (torch.Tensor): The reconstruction tensor of shape (x, y, z, time).
        time_idx (int): The time index to visualize.
        tr (float): The repetition time (TR) in seconds.
        n_subtract (int, optional): The number of previous time points to subtract for the subtraction image.
    """
    magnitude = torch.abs(recon)

    if n_subtract is not None and (time_idx - n_subtract >= 0):
        current_images = magnitude[:, :, :, time_idx]
        past_images = magnitude[:, :, :, time_idx - n_subtract]
        subtracted_images = current_images - past_images
    else:
        subtracted_images = magnitude[:, :, :, time_idx]

    # Maximum intensity projection across the z-axis
    mip = torch.max(subtracted_images, dim=2).values

    vmin, vmax = get_windowed_bounds(mip, lower_percentile, upper_percentile, window)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(mip.numpy(), cmap="gray", vmin=vmin, vmax=vmax, interpolation="none")
    ax.axis("off")
    ax.set_title(
        ("subtraction " if n_subtract is not None else "")
        + f"philips recon at {time_idx * tr:.1f}s (time point {time_idx}, TR {tr:.2f}s)"
    )
    if window is not None:
        ax.add_patch(
            patches.Rectangle(
                (0, 0),  # (x, y) position of the bottom left corner
                mip.shape[1],  # width of the rectangle
                window,  # height of the rectangle
                linewidth=1,
                edgecolor="r",
                facecolor="none",
                linestyle="--",
            )
        )


def interactive_plot(plot_fn, data, tr, start_value=0, **kwargs):
    """
    Create an interactive plot with a slider to cycle through timepoints, showing MRI coil images and Philips reconstruction in separate figures.
    Args:
        img_space (torch.Tensor): The input image space data of shape (x, y, z, coils, time).
        phillips_recon (torch.Tensor): Philips reconstruction tensor of shape (x, y, z, time).
        tr (float): The repetition time (TR) in seconds.
        n_subtract (int, optional): The number of time points to subtract for the subtraction image.
        window (int): The point to cut images to get rid of thorax/heart signal.
    """
    max_time_idx = data.shape[-1] - 1

    def update_plot(time_idx):
        clear_output(wait=True)
        plot_fn(data, time_idx, tr, **kwargs)

    time_slider = widgets.IntSlider(
        min=0, max=max_time_idx, step=1, value=start_value, description="time index"
    )
    interactive_control = widgets.interactive(update_plot, time_idx=time_slider)
    display(interactive_control)


def plot_histograms(signal_roi, noise_roi, title):
    plt.figure(figsize=(10, 5))
    plt.hist(signal_roi.numpy(), bins=30, alpha=0.7, label="signal ROI")
    plt.hist(noise_roi.numpy(), bins=30, alpha=0.7, label="noise ROI")
    plt.title(title)
    plt.xlabel("intensity")
    plt.ylabel("frequency")
    plt.legend()


def plot_rois(sense_img, ce_img, sense_rois, ce_rois, title):
    """
    Plot the ROIs on the SENSE image and CE image side by side.

    Parameters:
    sense_img (np.ndarray): The SENSE image.
    ce_img (np.ndarray): The CE image.
    sense_rois (list): List of ROIs in the SENSE image.
    ce_rois (list): List of ROIs in the CE image.
    title (str): Title for the plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Plot SENSE image with ROIs
    sense_img_with_rois = sense_img.copy()
    for x, y, w, h in sense_rois:
        cv2.rectangle(sense_img_with_rois, (x, y), (x + w, y + h), (255, 0, 0), 2)
    axes[0].imshow(sense_img_with_rois, cmap="gray")
    axes[0].set_title("SENSE Image with ROIs")

    # Plot CE image with ROIs
    ce_img_with_rois = ce_img.copy()
    for x, y, w, h in ce_rois:
        cv2.rectangle(ce_img_with_rois, (x, y), (x + w, y + h), (255, 0, 0), 2)
    axes[1].imshow(ce_img_with_rois, cmap="gray")
    axes[1].set_title("CE Image with ROIs")

    plt.suptitle(title)
