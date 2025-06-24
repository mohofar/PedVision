import numpy as np
import cv2
import matplotlib.pyplot as plt


def add_noise(image, noise_type, target_snr_percent):
    """
    Adds noise to an input image based on a target Signal-to-Noise Ratio (SNR).

    :param image: Input image (numpy array)
    :param noise_type: Type of noise ('gaussian', 'salt_pepper', 'motion')
    :param target_snr_percent: Target SNR as percentage (0-100%), where 0% is most noisy, 100% is clean
    :return: Noisy image (numpy array)
    """
    # Convert percentage to dB (0% -> ~0dB (very noisy), 100% -> ~40dB (almost clean))
    # Use a non-linear scale to better match perceptual quality
    if target_snr_percent <= 0:
        target_snr_db = 40  # Minimum noise
    elif target_snr_percent >= 100:
        target_snr_db = 0  # Maximum noise
    else:
        target_snr_db = 40 * (1-(target_snr_percent / 100))

    # Make a copy of the image and ensure it's float for calculations
    image_float = image.astype(np.float32)

    # Get image intensity range and signal power
    min_val, max_val = 0, 255  # Standard image range
    signal_power = np.mean(image_float ** 2)

    # Only compute noise if we're not at 100% SNR (clean image)
    if target_snr_percent < 100:
        noise_power = signal_power / (10 ** (target_snr_db / 10))  # Compute noise power from SNR
    else:
        noise_power = 0

    # Create a copy of the image to add noise to
    noisy_image = np.copy(image_float)

    if noise_type == 'gaussian':
        mean = 0
        stddev = np.sqrt(noise_power)
        gaussian_noise = np.random.normal(mean, stddev, image.shape).astype(np.float32)
        noisy_image = image_float + gaussian_noise

    elif noise_type == 'salt_pepper':
        s_vs_p = 0.5  # Salt vs Pepper ratio

        # Convert SNR to amount (density) - higher SNR means less salt and pepper
        # Map from 0dB (worst) to 40dB (best)
        amount = 0.05 * (1 - target_snr_db / 40)

        noisy_image = np.copy(image_float)

        # Generate coordinates for salt noise (separately for each channel if color image)
        if len(image.shape) == 3:  # Color image
            # Handle salt noise
            salt_mask = np.random.rand(*image.shape[:2]) < (amount * s_vs_p)
            salt_mask = np.stack([salt_mask] * image.shape[2], axis=2)
            noisy_image[salt_mask] = max_val

            # Handle pepper noise
            pepper_mask = np.random.rand(*image.shape[:2]) < (amount * (1 - s_vs_p))
            pepper_mask = np.stack([pepper_mask] * image.shape[2], axis=2)
            noisy_image[pepper_mask] = min_val
        else:  # Grayscale image
            # Handle salt noise
            salt_mask = np.random.rand(*image.shape) < (amount * s_vs_p)
            noisy_image[salt_mask] = max_val

            # Handle pepper noise
            pepper_mask = np.random.rand(*image.shape) < (amount * (1 - s_vs_p))
            noisy_image[pepper_mask] = min_val

    elif noise_type == 'motion':
        # Adjust kernel size based on SNR (inverse relationship - higher SNR, less blur)
        # Map from ~3 (at 40dB/100%) to ~31 (at 0dB/0%)
        size = max(3, int(31 - (target_snr_db / 40) * 28))
        # Make sure size is odd
        if size % 2 == 0:
            size += 1

        kernel = np.zeros((size, size))
        kernel[int((size - 1)/2), :] = np.ones(size)
        kernel /= size

        # Apply motion blur to each channel separately if needed
        if len(image.shape) == 3:  # Color image
            noisy_image = np.zeros_like(image_float)
            for i in range(image.shape[2]):
                noisy_image[:, :, i] = cv2.filter2D(image_float[:, :, i], -1, kernel)
        else:  # Grayscale image
            noisy_image = cv2.filter2D(image_float, -1, kernel)

    else:
        raise ValueError("Unsupported noise type. Choose from 'gaussian', 'salt_pepper', or 'motion'.")

    # Clip to valid range and convert back to uint8
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def show_noise_examples(image, noise_type, snr_levels):
    """
    Display original image alongside noisy versions at specified SNR levels.

    :param image: Input image (numpy array) - can be BGR or RGB
    :param noise_type: Type of noise ('gaussian', 'poisson', 'salt_pepper', 'speckle', 'motion')
    :param snr_levels: Single SNR value or list of SNR values (0-100)
    """
    # Ensure snr_levels is a list
    if isinstance(snr_levels, (int, float)):
        snr_levels = [snr_levels]

    # Convert BGR to RGB if needed (check if it's likely BGR format)
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Simple heuristic: if blue channel has higher mean than red, likely BGR
        if np.mean(image[:,:,0]) > np.mean(image[:,:,2]):
            display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            display_image = image
    else:
        display_image = image

    # Calculate number of subplots needed (original + noisy versions)
    num_plots = len(snr_levels) + 1

    # Determine subplot layout
    if num_plots <= 2:
        rows, cols = 1, num_plots
        figsize = (6 * num_plots, 5)
    elif num_plots <= 4:
        rows, cols = 1, num_plots
        figsize = (4 * num_plots, 4)
    elif num_plots <= 6:
        rows, cols = 2, 3
        figsize = (12, 8)
    else:
        rows = (num_plots + 2) // 3  # Ceiling division
        cols = 3
        figsize = (12, 4 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Handle single subplot case
    if num_plots == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if hasattr(axes, '__len__') else [axes]
    else:
        axes = axes.flatten()

    # Display original image
    axes[0].imshow(display_image)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Display noisy versions
    for i, snr in enumerate(snr_levels):
        try:
            noisy_img = add_noise(display_image, noise_type, snr)
            axes[i + 1].imshow(noisy_img)
            axes[i + 1].set_title(f'{noise_type.title()} Noise\nSNR: {snr}%', fontsize=11)
            axes[i + 1].axis('off')
        except Exception as e:
            axes[i + 1].text(0.5, 0.5, f'Error:\n{str(e)}',
                            ha='center', va='center', fontsize=10,
                            transform=axes[i + 1].transAxes)
            axes[i + 1].set_title(f'Error - SNR: {snr}%', fontsize=11)
            axes[i + 1].axis('off')

    # Hide unused subplots
    for j in range(num_plots, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    # plt.suptitle(f'{noise_type.title()} Noise Demonstration',
    #              fontsize=14, fontweight='bold', y=0.98)
    plt.show()

    # Print summary
    print(f"Noise type: {noise_type}")
    print(f"SNR levels tested: {snr_levels}")
    print(f"Image shape: {display_image.shape}")
    print(f"Lower SNR = More noise, Higher SNR = Less noise")
    print("=====================================================")
    
# Example usage functions for convenience
def quick_noise_demo(image, noise_type, light_heavy=True):
    """
    Quick demonstration with predefined light and heavy noise levels.

    :param image: Input image
    :param noise_type: Type of noise
    :param light_heavy: If True, shows light and heavy noise. If False, shows medium range.
    """
    if light_heavy:
        levels = [80, 30]  # Light noise, Heavy noise
        print("Showing light (80% SNR) and heavy (30% SNR) noise examples")
    else:
        levels = [70, 50, 30]  # Medium range
        print("Showing medium range noise examples")

    show_noise_examples(image, noise_type, levels)

def compare_all_noise_types(image, snr_level=50):
    """
    Compare all noise types at a single SNR level.

    :param image: Input image
    :param snr_level: SNR level to use for all noise types
    """
    noise_types = ['gaussian', 'salt_pepper', 'motion']

    # Convert BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        if np.mean(image[:,:,0]) > np.mean(image[:,:,2]):
            display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            display_image = image
    else:
        display_image = image

    fig, axes = plt.subplots(1, 4, figsize=(25, 20))
    axes = axes.flatten()

    # Original image
    axes[0].imshow(display_image)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Apply each noise type
    for i, noise_type in enumerate(noise_types):
        try:
            noisy_img = add_noise(display_image, noise_type, snr_level)
            axes[i + 1].imshow(noisy_img, 'gray')
            axes[i + 1].set_title(f'{noise_type.title()}\nSNR: {snr_level}%', fontsize=11)
            axes[i + 1].axis('off')
        except Exception as e:
            axes[i + 1].text(0.5, 0.5, f'Error:\n{str(e)[:30]}...',
                            ha='center', va='center', fontsize=9,
                            transform=axes[i + 1].transAxes)
            axes[i + 1].set_title(f'{noise_type.title()}\nError', fontsize=11)
            axes[i + 1].axis('off')

    plt.tight_layout()
    plt.show()

    print(f"Compared all noise types at {snr_level}% SNR")
    print("Noise types: Gaussian, Salt & Pepper, Motion")

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
def save_noise_image(image_path, save_path, noise_type, noise_levels, grayscale=True):
    """
    Generate and save noisy versions of an image.

    :param image_path: Path to the input image
    :param save_path: Directory path where noisy images will be saved
    :param noise_type: Type of noise ('gaussian', 'poisson', 'salt_pepper', 'speckle', 'motion')
    :param noise_levels: Single SNR value or list of SNR values (0-100)
    :param grayscale: If True, saves as grayscale; if False, saves as color
    """
    # Ensure noise_levels is a list
    if isinstance(noise_levels, (int, float)):
        noise_levels = [noise_levels]

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Convert BGR to RGB for processing
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get base name without extension
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    print(f"Processing image: {base_name}")
    print(f"Noise type: {noise_type}")
    print(f"SNR levels: {noise_levels}")
    print(f"Grayscale output: {grayscale}")

    saved_files = []

    # Generate and save noisy images
    for level in noise_levels:
        try:
            # Generate noisy image
            noisy_img = add_noise(image_rgb, noise_type, level)

            # Convert to grayscale if requested
            if grayscale:
                noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_RGB2GRAY)
                filename = f"{base_name}_{noise_type}_N_lvl_{level}.jpg"
            else:
                # Convert RGB back to BGR for OpenCV saving
                noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_RGB2BGR)
                filename = f"{base_name}_{noise_type}_N_lvl_{level}_color.jpg"

            # Full path for saving
            full_path = os.path.join(save_path, filename)

            # Save the image
            success = cv2.imwrite(full_path, noisy_img)

            if success:
                saved_files.append(full_path)
                print(f"✓ Saved: {filename}")
            else:
                print(f"✗ Failed to save: {filename}")

        except Exception as e:
            print(f"✗ Error processing SNR level {level}: {str(e)}")

    print(f"\nCompleted! Saved {len(saved_files)} images to {save_path}")
    return saved_files

def visualize_masks(img_path, cls_num=5):
    """
    Visualizes predicted masks for each class from VFM-based model.

    Args:
        image (numpy.ndarray): The original image.
        img_path (str): The base name of the image file (without extension).
        cls_num (int): The number of classes to visualize.
    """
    image = cv2.imread(img_path)

    img_name = os.path.basename(img_path)[:-4]
    print(f"Visualizing masks for: {img_name}")

    mask_path = f'PedVisionCode/test_data/predicted/VFM/org_mask_{img_name}.pkl'
    try:
        with open(mask_path, 'rb') as f:
            masks = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Mask file not found at {mask_path}")
        return
    except Exception as e:
        print(f"Error loading mask file: {e}")
        return

    prediction_path = f'PedVisionCode/test_data/prepared/{img_name}.npy'
    try:
        prediction = np.load(prediction_path)
    except FileNotFoundError:
        print(f"Error: Prediction file not found at {prediction_path}")
        return
    except Exception as e:
        print(f"Error loading prediction file: {e}")
        return

    # Visualize predicted masks
    plt.figure(figsize=(30, 30))
    # Display each class's mask
    for bone in range(cls_num):
        overall_mask = np.zeros(image.shape[:2])
        for i, pred in enumerate(prediction):
            if pred.item() == bone:
                # Ensure masks[i]['crop_box'] and masks[i]['segmentation'] are valid
                try:
                    crop_box = masks[i]['crop_box']
                    segmentation = masks[i]['segmentation']
                    if crop_box is not None and segmentation is not None:
                        x1, x2, y1, y2 = crop_box
                        # Resize segmentation mask to match the crop box dimensions if necessary
                        # (This might be needed depending on how the segmentation is stored)
                        # For now, assuming segmentation matches the crop box size
                        if segmentation.shape == (x2 - x1, y2 - y1):
                            overall_mask[x1:x2, y1:y2] += segmentation
                        else:
                             # If shapes don't match, resize the segmentation mask
                             resized_segmentation = cv2.resize(segmentation.astype(np.uint8), (y2 - y1, x2 - x1), interpolation=cv2.INTER_NEAREST)
                             overall_mask[x1:x2, y1:y2] += resized_segmentation


                except (IndexError, KeyError) as e:
                    print(f"Warning: Skipping mask {i} due to missing key or invalid index: {e}")
                except Exception as e:
                     print(f"Warning: Skipping mask {i} due to processing error: {e}")


        plt.subplot(1, cls_num, bone + 1)
        plt.imshow(image)
        plt.imshow(overall_mask.astype(np.bool_), alpha=0.7)
        plt.title(f'Class {bone}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()
