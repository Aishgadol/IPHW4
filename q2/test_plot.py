# zebra.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift, ifft2, ifftshift

def normalize_spectrum_log(spectrum):
    """
    Normalize the magnitude spectrum using logarithmic scaling to the range 0-255.
    """
    magnitude_spectrum = np.log(1 + np.abs(spectrum))
    magnitude_spectrum = (magnitude_spectrum / magnitude_spectrum.max()) * 255
    return magnitude_spectrum.astype(np.uint8)

def apply_clahe(spectrum, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance the spectrum.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_spectrum = clahe.apply(spectrum)
    return enhanced_spectrum

def main():
    # 1. Load the Original Grayscale Image
    image_path = 'zebra.jpg'
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if gray_image is None:
        print(f"Error: Unable to load image at path '{image_path}'. Please check the file path.")
        return

    H, W = gray_image.shape
    print(f"Loaded image with dimensions: Height={H}, Width={W}")

    # 2. Perform FFT on Original Image
    fourier_original = fft2(gray_image)
    fourier_original_shifted = fftshift(fourier_original)
    magnitude_original_log = normalize_spectrum_log(fourier_original_shifted)

    # Apply CLAHE to the original Fourier spectrum for enhanced visualization
    magnitude_original_clahe = apply_clahe(magnitude_original_log)

    # 3. Zero-Pad the Image to (2H, 2W)
    pad_H_before = H // 2
    pad_H_after = H - pad_H_before
    pad_W_before = W // 2
    pad_W_after = W - pad_W_before

    padded_image = np.pad(gray_image,
                          ((pad_H_before, pad_H_after), (pad_W_before, pad_W_after)),
                          mode='constant', constant_values=0)

    # Perform FFT on Padded Image
    fourier_padded = fft2(padded_image)
    fourier_padded_shifted = fftshift(fourier_padded)
    magnitude_padded_log = normalize_spectrum_log(fourier_padded_shifted)

    # Apply CLAHE to the padded Fourier spectrum for enhanced visualization
    magnitude_padded_clahe = apply_clahe(magnitude_padded_log)

    # 4. Zero-Inserted Fourier Spectrum and Restored Scaled Image
    zero_inserted_fourier = np.zeros((2 * H, 2 * W), dtype=complex)
    zero_inserted_fourier[::2, ::2] = fourier_original_shifted
    zero_inserted_fourier_shifted = ifftshift(zero_inserted_fourier)
    restored_scaled_image = np.abs(ifft2(zero_inserted_fourier_shifted))
    # Normalize restored image for display
    restored_scaled_image_normalized = cv2.normalize(restored_scaled_image, None, 0, 255, cv2.NORM_MINMAX)
    restored_scaled_image_normalized = restored_scaled_image_normalized.astype(np.uint8)

    # Compute Magnitude Spectrum of Zero-Inserted Fourier Spectrum
    magnitude_zero_inserted_log = normalize_spectrum_log(zero_inserted_fourier)
    magnitude_zero_inserted_clahe = apply_clahe(magnitude_zero_inserted_log)

    # 5. Plot Configuration
    fig, axes = plt.subplots(4, 2, figsize=(30, 80), constrained_layout=True)

    # 1a) Original Grayscale Image
    axes[0, 0].imshow(gray_image, cmap='gray')
    axes[0, 0].axis('off')
    axes[0, 0].set_title('1a) Original Grayscale Image', fontsize=20)

    # 1b) Fourier Spectrum of Original Image (Log Normalized)
    axes[0, 1].imshow(magnitude_original_log, cmap='gray')
    axes[0, 1].axis('off')
    axes[0, 1].set_title('1b) Fourier Spectrum of Original Image (Log Normalized)', fontsize=20)

    # Optional: 1c) Fourier Spectrum of Original Image (CLAHE Enhanced)
    # Uncomment the following lines if you want to include CLAHE-enhanced spectra
    # axes[0, 2].imshow(magnitude_original_clahe, cmap='gray')
    # axes[0, 2].axis('off')
    # axes[0, 2].set_title('1c) Fourier Spectrum of Original Image (CLAHE Enhanced)', fontsize=20)

    # 2a) Padded Grayscale Image (2H x 2W)
    axes[1, 0].imshow(padded_image, cmap='gray')
    axes[1, 0].axis('off')
    axes[1, 0].set_title('2a) Padded Grayscale Image (2H x 2W)', fontsize=20)

    # 2b) Fourier Spectrum of Padded Image (Log Normalized)
    axes[1, 1].imshow(magnitude_padded_log, cmap='gray')
    axes[1, 1].axis('off')
    axes[1, 1].set_title('2b) Fourier Spectrum of Padded Image (Log Normalized)', fontsize=20)

    # 3a) Restored Scaled Image (Four Copies)
    axes[2, 0].imshow(restored_scaled_image_normalized, cmap='gray')
    axes[2, 0].axis('off')
    axes[2, 0].set_title('3a) Restored Scaled Image (Four Copies)', fontsize=20)

    # 3b) Zero-Inserted Fourier Spectrum (Log Normalized)
    axes[2, 1].imshow(magnitude_zero_inserted_log, cmap='gray')
    axes[2, 1].axis('off')
    axes[2, 1].set_title('3b) Zero-Inserted Fourier Spectrum (Log Normalized)', fontsize=20)

    # 4a) Restored Scaled Image (Four Copies) - Repeated for Comparison
    axes[3, 0].imshow(restored_scaled_image_normalized, cmap='gray')
    axes[3, 0].axis('off')
    axes[3, 0].set_title('4a) Restored Scaled Image (Four Copies)', fontsize=20)

    # 4b) Zero-Inserted Fourier Spectrum (CLAHE Enhanced)
    axes[3, 1].imshow(magnitude_zero_inserted_clahe, cmap='gray')
    axes[3, 1].axis('off')
    axes[3, 1].set_title('4b) Zero-Inserted Fourier Spectrum (CLAHE Enhanced)', fontsize=20)

    # 6. Display and Save the Plot
    plt.show()

    # To save the figure with high resolution, uncomment the following line:
    # plt.savefig('zebra_plots.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()
