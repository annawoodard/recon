import numpy as np
import cv2
import SimpleITK as sitk
import torch
from plot import plot_histograms, plot_rois


def normalize_image(image):
    """
    Normalize the image to have zero mean and unit variance.

    Parameters:
    image (np.ndarray): The input image.

    Returns:
    np.ndarray: The normalized image.
    """
    mean = np.mean(image)
    std = np.std(image)
    normalized_image = (image - mean) / std
    return normalized_image


def register_images(fixed_image, moving_image):
    """
    Register two images using SimpleITK.

    Parameters:
    fixed_image (np.ndarray): The fixed image (e.g., SENSE image).
    moving_image (np.ndarray): The moving image to be aligned (e.g., CE image).

    Returns:
    np.ndarray: The registered moving image and the transformation used.
    """
    # Ensure both images are 2D and of the same type
    fixed_image, moving_image = center_images(fixed_image, moving_image)
    assert np.sum(fixed_image) != 0, "Fixed image is empty."
    assert np.sum(moving_image) != 0, "Moving image is empty."

    fixed_image_sitk = sitk.GetImageFromArray(fixed_image.astype(np.float32))
    moving_image_sitk = sitk.GetImageFromArray(moving_image.astype(np.float32))

    # Ensure both images have the same number of dimensions
    assert (
        fixed_image_sitk.GetDimension() == moving_image_sitk.GetDimension()
    ), "Fixed and moving images must have the same number of dimensions."

    # Initialize registration method
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=200
    )

    # Set initial transform
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image_sitk, moving_image_sitk, sitk.Euler2DTransform()
    )
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Execute registration
    final_transform = registration_method.Execute(
        sitk.Cast(fixed_image_sitk, sitk.sitkFloat32),
        sitk.Cast(moving_image_sitk, sitk.sitkFloat32),
    )
    moving_resampled = sitk.Resample(
        moving_image_sitk,
        fixed_image_sitk,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving_image_sitk.GetPixelID(),
    )

    return sitk.GetArrayFromImage(moving_resampled), final_transform


def apply_transform(image, transform):
    """
    Apply a transform to an image using SimpleITK.

    Parameters:
    image (np.ndarray): The input image.
    transform (SimpleITK.Transform): The transform to apply.

    Returns:
    np.ndarray: The transformed image.
    """
    image_sitk = sitk.GetImageFromArray(image)
    reference_image = sitk.GetImageFromArray(image)
    transformed_image = sitk.Resample(
        image_sitk,
        reference_image,
        transform,
        sitk.sitkLinear,
        0.0,
        image_sitk.GetPixelID(),
    )
    return sitk.GetArrayFromImage(transformed_image)


def detect_enhancing_features(image, threshold=150):
    """
    Detect enhancing features in an image using thresholding and edge detection.

    Parameters:
    image (np.ndarray): The input image.
    threshold (int): Intensity threshold for detecting enhancing features.

    Returns:
    np.ndarray: Binary mask of the enhancing features.
    """
    # Ensure the image is in the correct format for Canny
    image = (image * 255).astype(np.uint8)

    # Apply thresholding
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    # Ensure the binary image is of type CV_8U
    binary_image = binary_image.astype(np.uint8)

    # Apply Canny edge detection
    edges = cv2.Canny(binary_image, 100, 200)

    # Dilate edges
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    return dilated_edges


def max_intensity_projection(image, axis=2):
    return np.max(image, axis=axis)


def extract_rois(image, mask):
    """
    Extract ROIs from an image using a binary mask.

    Parameters:
    image (np.ndarray): The input image.
    mask (np.ndarray): The binary mask of the ROIs.

    Returns:
    list: List of bounding box coordinates for each ROI.
    """
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract bounding boxes
    rois = [cv2.boundingRect(contour) for contour in contours]

    return rois


def window_image(image, window=None):
    if window is not None:
        # Ensure window does not exceed image dimensions
        window = min(window, image.shape[0])
        windowed_image = image[:window, ...]
    else:
        windowed_image = image

    return windowed_image


def get_subtraction_image(image, timepoint, n_subtract=None):
    if n_subtract is not None and (timepoint - n_subtract >= 0):
        current_image = image[..., timepoint]
        past_image = image[..., timepoint - n_subtract]
        return current_image - past_image
    return image[..., timepoint]


def center_images(fixed_image, moving_image):
    """
    Align the centers of the fixed and moving images by translating the moving image's center to match the fixed image's center.

    Parameters:
    fixed_image (np.ndarray): The fixed image.
    moving_image (np.ndarray): The moving image to be aligned.

    Returns:
    np.ndarray, np.ndarray: The centered moving image.
    """
    # Convert numpy arrays to SimpleITK images
    fixed_image_sitk = sitk.GetImageFromArray(
        fixed_image.astype(np.float32), isVector=False
    )
    moving_image_sitk = sitk.GetImageFromArray(
        moving_image.astype(np.float32), isVector=False
    )

    # Set the spacing, origin, and size for the moving image to match the fixed image
    moving_image_sitk.SetSpacing(fixed_image_sitk.GetSpacing())
    moving_image_sitk.SetOrigin(fixed_image_sitk.GetOrigin())

    # Calculate centers in physical space
    fixed_center = np.array(
        fixed_image_sitk.TransformContinuousIndexToPhysicalPoint(
            np.array(fixed_image_sitk.GetSize()) / 2.0
        )
    )
    moving_center = np.array(
        moving_image_sitk.TransformContinuousIndexToPhysicalPoint(
            np.array(moving_image_sitk.GetSize()) / 2.0
        )
    )
    print("fixed center is:", fixed_center)
    print("moving center is:", moving_center)

    # Calculate the translation needed to align centers
    translation = np.array(fixed_center) - np.array(moving_center)
    translation_transform = sitk.TranslationTransform(fixed_image_sitk.GetDimension())
    translation_transform.SetOffset(translation)

    # Use a resampler to align the moving image to the fixed image
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image_sitk)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(translation_transform)
    resampled_moving_image = resampler.Execute(moving_image_sitk)
    assert (
        np.sum(sitk.GetArrayFromImage(moving_image_sitk)) != 0
    ), "Foo Moving image is empty."
    assert (
        np.sum(sitk.GetArrayFromImage(resampled_moving_image)) != 0
    ), "Resampled moving image is empty."

    # Check for empty image (all zero)
    if np.all(sitk.GetArrayFromImage(resampled_moving_image) == 0):
        print(
            "Warning: The resampled moving image is empty. Check the transformation settings."
        )

    return fixed_image, sitk.GetArrayFromImage(resampled_moving_image)


def process_images(
    coil_images,
    sense_image,
    ce,
    timepoint,
    threshold=150,
    coil_window=None,
    recon_window=None,
    n_subtract=None,
):
    """
    Complete pipeline to process images, perform registration, feature detection, ROI extraction, and SNR calculation.

    Parameters:
    coil_images (torch.Tensor): Tensor of coil images with shape (kx, ky, kz, num_coil, time).
    sense_image (torch.Tensor): Tensor of SENSE images with shape (x, y, z, time).
    ce (int): Coil element index to match.
    timepoint (int): Time point to match.
    threshold (int): Intensity threshold for detecting enhancing features.

    Returns:
    None
    """
    coil_images = window_image(coil_images, coil_window)
    sense_image = window_image(sense_image, recon_window)

    coil_images = np.abs(coil_images)
    sense_image = np.abs(sense_image)

    # coil_images = get_subtraction_image(coil_images, timepoint, n_subtract)
    coil_images = coil_images[:, :, :, :, timepoint]
    sense_image = get_subtraction_image(sense_image, timepoint, n_subtract)

    coil_images = coil_images[:, :, :, ce].numpy()
    sense_image = sense_image.numpy()

    coil_images = max_intensity_projection(coil_images)
    sense_image = max_intensity_projection(sense_image)

    coil_images = normalize_image(coil_images)
    sense_image = normalize_image(sense_image)

    registered_image, transform = register_images(sense_image, coil_images)

    # Detect and extract ROIs in the registered image
    enhancing_features = detect_enhancing_features(registered_image, threshold)
    rois = extract_rois(registered_image, enhancing_features)

    # Apply the same transformation to the CE image to get corresponding ROIs
    ce_transformed_image = apply_transform(coil_images, transform)

    # Plot ROIs in both SENSE and CE images
    # plot_rois(
    #     sense_image,
    #     coil_images,
    #     [],
    #     [],
    #     "Registered ROIs in SENSE and CE Images",
    # )
    plot_rois(
        sense_image,
        ce_transformed_image,
        rois,
        rois,
        "Registered ROIs in SENSE and CE Images",
    )

    # Flatten and calculate SNR for corresponding ROIs
    sense_roi_values = [
        sense_image[y : y + h, x : x + w].flatten() for x, y, w, h in rois
    ]
    ce_roi_values = [
        ce_transformed_image[y : y + h, x : x + w].flatten() for x, y, w, h in rois
    ]

    sense_signal_roi_tensor = torch.tensor(np.concatenate(sense_roi_values))
    ce_signal_roi_tensor = torch.tensor(np.concatenate(ce_roi_values))
    noise_roi_tensor = torch.tensor(registered_image.flatten())

    sense_snr = calculate_snr(sense_signal_roi_tensor, noise_roi_tensor)
    ce_snr = calculate_snr(ce_signal_roi_tensor, noise_roi_tensor)

    print(f"SNR for SENSE image: {sense_snr}")
    print(f"SNR for CE image: {ce_snr}")

    plot_histograms(sense_signal_roi_tensor, noise_roi_tensor, "SENSE image ROIs")
    plot_histograms(ce_signal_roi_tensor, noise_roi_tensor, "CE image ROIs")


def calculate_snr(signal_roi, noise_roi):
    """
    Calculate the SNR given signal and noise ROIs.

    Parameters:
    signal_roi (torch.Tensor): Tensor containing the signal ROI intensities.
    noise_roi (torch.Tensor): Tensor containing the noise ROI intensities.

    Returns:
    float: The calculated SNR.
    """
    mu_signal = torch.mean(signal_roi)
    sigma_noise = torch.std(noise_roi)
    snr = mu_signal / sigma_noise
    return snr.item()
