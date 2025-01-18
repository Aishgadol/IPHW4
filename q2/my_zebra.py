#zebra.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift, ifft2, ifftshift

def normalize_spectrum_log(spectrum):
    magnitude = np.log1p(np.abs(spectrum))  #log1p is log(1 + x)
    magnitude = (magnitude / np.max(magnitude)) * 255
    return magnitude.astype(np.uint8)

def pad_image_normal(image, target_shape):
    original_h, original_w = image.shape
    target_h, target_w = target_shape

    #calculate padding sizes
    pad_h_before = (target_h - original_h) // 2
    pad_h_after = target_h - original_h - pad_h_before
    pad_w_before = (target_w - original_w) // 2
    pad_w_after = target_w - original_w - pad_w_before

    #apply normal zero-padding
    padded = np.pad(
        image,
        ((pad_h_before, pad_h_after), (pad_w_before, pad_w_after)),
        mode='constant',
        constant_values=0
    )

    return padded

def pad_fourier_spectrum_inserted(fourier_shifted):
    original_h, original_w = fourier_shifted.shape
    target_h, target_w = original_h * 2, original_w * 2

    #initialize a zero-filled spectrum
    inserted_padded = np.zeros((target_h, target_w), dtype=complex)

    #assign original fourier coefficients to even indices
    inserted_padded[::2, ::2] = fourier_shifted

    return inserted_padded

def main():
    #1. load the original grayscale image
    image_path = 'zebra.jpg'
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if gray is None:
        print(f"error loading image")
        return

    h, w = gray.shape

    #2. perform fft on original image
    fourier_orig = fft2(gray)
    fourier_orig_shifted = fftshift(fourier_orig)
    magnitude_orig_log = normalize_spectrum_log(fourier_orig_shifted)

    #3. normal zero-padding on image (spatial domain)
    target_h_normal, target_w_normal = 2 * h, 2 * w
    padded_normal = pad_image_normal(gray, (target_h_normal, target_w_normal))

    #4. perform fft on normally padded image
    fourier_padded_normal = fft2(padded_normal)
    fourier_padded_normal_shifted = fftshift(fourier_padded_normal)
    magnitude_padded_normal_log = normalize_spectrum_log(fourier_padded_normal_shifted)

    #5. inserted zero-padding on fourier spectrum
    inserted_padded_shifted = pad_fourier_spectrum_inserted(fourier_orig_shifted)
    magnitude_inserted_log = normalize_spectrum_log(inserted_padded_shifted)

    #6. inverse fft of inserted-padded fourier spectrum
    inserted_padded = ifftshift(inserted_padded_shifted)
    restored_inserted = ifft2(inserted_padded)
    restored_inserted = np.abs(restored_inserted)
    restored_inserted_norm = cv2.normalize(
        restored_inserted, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    #plottings using the original code and some upgrades
    fig, axes = plt.subplots(3, 2, figsize=(20, 30), constrained_layout=True)

    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].axis('off')
    axes[0, 0].set_title('original grayscale image', fontsize=16)

    axes[0, 1].imshow(magnitude_orig_log, cmap='gray')
    axes[0, 1].axis('off')
    axes[0, 1].set_title('fourier spectrum', fontsize=16)

    axes[1, 0].imshow(padded_normal, cmap='gray')
    axes[1, 0].axis('off')
    axes[1, 0].set_title('two times larger grayscale image', fontsize=16)

    axes[1, 1].imshow(magnitude_padded_normal_log, cmap='gray')
    axes[1, 1].axis('off')
    axes[1, 1].set_title('fourier spectrum zero padding', fontsize=16)

    axes[2, 0].imshow(restored_inserted_norm, cmap='gray')
    axes[2, 0].axis('off')
    axes[2, 0].set_title('four copies grayscale image', fontsize=16)

    axes[2, 1].imshow(magnitude_inserted_log, cmap='gray')
    axes[2, 1].axis('off')
    axes[2, 1].set_title('fourier spectrum four copies', fontsize=16)

    plt.savefig('zebra_scaled.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
