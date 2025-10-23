import time
import cv2
import numpy as np
from scipy.stats import entropy


def sobel(image_in):
    t0 = time.time()

    gray = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)
    sobel_image = cv2.Sobel(gray, ddepth=cv2.CV_16S, dx=1, dy=1, ksize=3)
    sobel_value = sobel_image.var()
    image_out = sobel_image
    image_out = cv2.convertScaleAbs(sobel_image)
    image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2RGB)

    t1 = time.time()
    # print('Sobel Time:', t1-t0)

    return sobel_value, image_out


def sobel_cuda(image_in):
    t0 = time.time()

    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(image_in)
    gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)

    t1 = time.time()
    print('Data Transfer to GPU Time:', t1-t0)

    # Create and apply Sobel Filter
    sobel_filter = cv2.cuda.createSobelFilter(
        cv2.CV_8UC1, cv2.CV_16S, 1, 1, 3)
    gpu_sobel_image = sobel_filter.apply(gpu_gray)

    t2 = time.time()
    print('CUDA Kernel Execution Time:', t2-t1)

    sobel_image = gpu_sobel_image.download()

    t3 = time.time()
    print('Data Transfer from GPU Time:', t3-t2)

    sobel_value = sobel_image.var()
    image_out = sobel_image
    image_out = cv2.convertScaleAbs(sobel_image)
    image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2RGB)

    t4 = time.time()
    print('Total CUDA Sobel Time:', t4-t0)

    return sobel_value, image_out

# def sobel_cuda(image_in):
#     t0 = time.time()

#     gpu_frame = cv2.cuda_GpuMat()
#     gpu_frame.upload(image_in)
#     gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)

#     # Create and apply Sobel Filter
#     sobel_filter = cv2.cuda.createSobelFilter(
#         cv2.CV_8UC1, cv2.CV_16S, 1, 1, 3)
#     gpu_sobel_image = sobel_filter.apply(gpu_gray)
#     sobel_image = gpu_sobel_image.download()
#     sobel_value = sobel_image.var()
#     image_out = sobel_image
#     image_out = cv2.convertScaleAbs(sobel_image)
#     image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2RGB)

#     t1 = time.time()
#     print('CUDA Sobel Time:', t1-t0)

#     return sobel_value, image_out


def squared_gradient(image_in):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # Compute finite differences
    gradient_x = np.diff(gray_image, axis=1)  # shape: (H, W-1)
    gradient_y = np.diff(gray_image, axis=0)  # shape: (H-1, W)

    # Crop to matching dimensions
    gradient_x = gradient_x[:-1, :]  # (H-1, W-1)
    gradient_y = gradient_y[:, :-1]  # (H-1, W-1)

    # Squared gradient
    squared_gradient = gradient_x**2 + gradient_y**2

    # Focus metric: typically variance or sum of squared gradient
    focus_value = np.var(squared_gradient)  # or np.sum(squared_gradient)

    # Visualization (gradient magnitude)
    gradient_magnitude = np.sqrt(squared_gradient).astype(np.float32)
    normalized_image = cv2.normalize(
        gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image_out = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2RGB)

    return focus_value, image_out


def squared_sobel(image_in):
    gray_image = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)

    # Compute the squared differences of adjacent pixels in both directions using Sobel
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    signal = np.mean(gradient_magnitude)
    noise = np.std(gradient_magnitude)
    snr = signal/noise

    # np.var(smoothed_combined_gradient[yl:yh, xl:xh]) #+ np.mean(smoothed_combined_gradient[yl:yh, xl:xh])**1.5
    focus_value = np.var(gradient_magnitude)

    # normalized_image = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # image_out = cv2.convertScaleAbs(gradient_magnitude)
    # image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2RGB)

    normalized_image = cv2.normalize(
        gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image_out = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2RGB)
    return focus_value, image_out


# def fswm(image_in):
#     gray_image = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)
#     # ksize = 17
#     # sigma = 1.5

#     # # Apply median filter in x and y directions
#     # median_filtered_x = cv2.medianBlur(gray_image, ksize)
#     # median_filtered_y = cv2.medianBlur(gray_image.T, ksize).T

#     # fswm_x = np.abs(gray_image - median_filtered_x)
#     # fswm_y = np.abs(gray_image - median_filtered_y)
#     # combined_fswm = fswm_x + fswm_y

#     # # Apply Gaussian blur to denoise
#     # denoised_combined_fswm = cv2.GaussianBlur(
#     #     combined_fswm, (0, 0), sigmaX=sigma, sigmaY=sigma)

#     # focus_value = np.var(denoised_combined_fswm)

#     # normalized_image = cv2.normalize(
#     #     denoised_combined_fswm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#     # image_out = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2RGB)[
#     #     y0:y1, x0:x1]

#     # Apply a bandpass filter using Difference of Gaussians (DoG)
#     sigma_low = 2.5
#     sigma_high = 3.0
#     blur_low = cv2.GaussianBlur(gray_image, (0, 0), sigmaX=sigma_low)
#     blur_high = cv2.GaussianBlur(gray_image, (0, 0), sigmaX=sigma_high)
#     bandpass = blur_low - blur_high

#     # Create a weight matrix
#     rows, cols = bandpass.shape
#     center_y, center_x = rows // 2, cols // 2
#     Y, X = np.ogrid[:rows, :cols]
#     distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
#     max_distance = np.max(distance)
#     # Weights decrease with distance from center
#     weights = 1 - (distance / max_distance)

#     # Compute the weighted mean
#     weighted_bandpass = bandpass * weights
#     focus_value = np.var(bandpass)

#     # For visualization, normalize the weighted bandpass image
#     bandpass_normalized = cv2.normalize(
#         weighted_bandpass, None, 0, 255, cv2.NORM_MINMAX)
#     image_out = cv2.cvtColor(bandpass_normalized.astype(
#         np.uint8), cv2.COLOR_GRAY2RGB)

#     return focus_value, image_out

def fswm(image_in):
    gray_image = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # FSWM parameters
    ksize = 5  # Median filter kernel size (must be odd)

    # Apply median filter (acts as low-pass filter)
    median_filtered = cv2.medianBlur(
        gray_image.astype(np.uint8), ksize).astype(np.float64)

    # High-pass filter: original - median (extracts high-frequency content)
    high_freq = np.abs(gray_image - median_filtered)

    # Optional: Apply center weighting (emphasizes center of image)
    rows, cols = high_freq.shape
    center_y, center_x = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    max_distance = np.sqrt(center_x**2 + center_y**2)

    # Gaussian-like weighting (higher weight at center)
    sigma_weight = max_distance / 3
    weights = np.exp(-(distance**2) / (2 * sigma_weight**2))

    # Apply weights to high-frequency content
    weighted_high_freq = high_freq * weights

    # Focus metric: sum or variance of weighted high-frequency content
    focus_value = np.sum(weighted_high_freq)  # or np.var(high_freq)

    # For visualization
    normalized_image = cv2.normalize(
        high_freq, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image_out = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2RGB)

    return focus_value, image_out


def fft(image_in):
    # # Convert the image to grayscale
    # gray_image = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)
    # size = 30
    # # Apply FFT to the entire grayscale image
    # f = np.fft.fft2(gray_image)
    # fshift = np.fft.fftshift(f)

    # # Logarithmic scaling for better visualization of the magnitude spectrum
    # magnitude_spectrum = 20 * np.log1p(np.abs(fshift))

    # # Determine the center of the frequency spectrum
    # rows, cols = gray_image.shape
    # cX, cY = cols // 2, rows // 2

    # # Zero out the low-frequency components around the center
    # fshift[cY - size:cY + size, cX - size:cX + size] = 0

    # # Apply the inverse FFT to focus on high-frequency components
    # f_ishift = np.fft.ifftshift(fshift)
    # recon = np.fft.ifft2(f_ishift)
    # recon = np.abs(recon)

    # # Calculate the focus value as the variance of the high-frequency components in the ROI
    # focus_value = np.var(recon)

    # # Normalize the magnitude spectrum for visualization
    # normalized_spectrum = cv2.normalize(
    #     magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # image_out = cv2.cvtColor(normalized_spectrum, cv2.COLOR_GRAY2RGB)

    gray = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)
    # Apply a window function to reduce edge effects
    window = np.hanning(gray.shape[0])[
        :, None] * np.hanning(gray.shape[1])[None, :]
    gray_windowed = gray * window
    # Compute the FFT
    f = np.fft.fft2(gray_windowed)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)

    # Ground zero low frequencies
    center_y, center_x = magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2
    low_freq_size = 10
    magnitude_spectrum[center_y - low_freq_size:center_y + low_freq_size,
                       center_x - low_freq_size:center_x + low_freq_size] = 0
    # Focus measure: sum of magnitude spectrum values
    focus_value = np.var(magnitude_spectrum)
    magnitude_spectrum_log = 20 * np.log1p(magnitude_spectrum)
    image_out = cv2.normalize(
        magnitude_spectrum_log, None, 0, 255, cv2.NORM_MINMAX)
    image_out = cv2.cvtColor(image_out.astype(
        np.uint8), cv2.COLOR_GRAY2BGR)
    return focus_value, image_out


def mix_sobel(image_in):
    gray_image = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)

    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_xy = cv2.Sobel(gray_image, cv2.CV_64F, 1, 1, ksize=3)
    combined_gradients = gradient_magnitude + np.abs(sobel_xy)
    focus_value = np.var(combined_gradients)

    normalized_image = cv2.normalize(
        combined_gradients, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image_out = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2RGB)

    return focus_value, image_out


def sobel_laplacian(image_in):
    # Convert to grayscale
    gray = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)

    # Apply Sobel filter
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Apply Laplacian filter
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Combine Sobel and Laplacian
    combined = sobel_magnitude + np.abs(laplacian)

    # Compute focus value
    focus_value = np.var(combined)

    # Normalize for visualization
    normalized_image = cv2.normalize(
        combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image_out = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2RGB)

    return focus_value, image_out


def wavelet(image_in):
    gray_image = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)
    roi = gray_image

    rows, cols = roi.shape
    if rows % 2 != 0:
        roi = roi[:-1, :]
        rows -= 1
    if cols % 2 != 0:
        roi = roi[:, :-1]
        cols -= 1

    # Perform single-level Haar wavelet transform manually
    LL = (roi[0::2, 0::2] + roi[0::2, 1::2] +
          roi[1::2, 0::2] + roi[1::2, 1::2]) / 4
    LH = (roi[0::2, 0::2] - roi[0::2, 1::2] +
          roi[1::2, 0::2] - roi[1::2, 1::2]) / 4
    HL = (roi[0::2, 0::2] + roi[0::2, 1::2] -
          roi[1::2, 0::2] - roi[1::2, 1::2]) / 4
    HH = (roi[0::2, 0::2] - roi[0::2, 1::2] -
          roi[1::2, 0::2] + roi[1::2, 1::2]) / 4

    # Calculate the energy of the high-frequency components
    high_freq = np.sqrt(LH**2 + HL**2 + HH**2 - LL**2)
    focus_value = np.mean(LH**2 + HL**2 + HH**2)
    high_freq_resized = cv2.resize(
        high_freq, (cols, rows), interpolation=cv2.INTER_LINEAR)

    # Normalize the image for display
    high_freq_normalized = cv2.normalize(
        high_freq_resized, None, 0, 255, cv2.NORM_MINMAX)
    image_out = cv2.cvtColor(high_freq_normalized.astype(
        np.uint8), cv2.COLOR_GRAY2RGB)

    return focus_value, image_out


def lpq(image_in):
    gray_image = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)
    roi = gray_image

    # Parameters
    win_size = 7
    rho = 0.95
    STFTalpha = 1.0 / win_size

    x = np.arange(-(win_size // 2), win_size // 2 + 1)
    wx = np.hamming(win_size)
    [X, Y] = np.meshgrid(x, x)

    w0 = (1 / win_size) * np.ones((win_size, win_size))
    w1 = np.exp(-2j * np.pi * STFTalpha * X)
    w2 = np.exp(-2j * np.pi * STFTalpha * Y)

    filters = [
        w0,
        w1,
        w2,
        w1 * w2
    ]

    LPQdesc = np.zeros(roi.shape, dtype=np.uint8)

    for i, filt in enumerate(filters[1:]):
        conv_real = cv2.filter2D(roi.astype(np.float32), -1, np.real(filt))
        conv_imag = cv2.filter2D(roi.astype(np.float32), -1, np.imag(filt))

        LPQdesc += ((conv_real >= 0).astype(np.uint8) << (2 * i))
        LPQdesc += ((conv_imag >= 0).astype(np.uint8) << (2 * i + 1))

    hist, _ = np.histogram(LPQdesc.ravel(), bins=256, range=(0, 256))
    focus_value = entropy(hist + np.finfo(float).eps)

    image_out = cv2.normalize(LPQdesc.astype(
        np.float32), None, 0, 255, cv2.NORM_MINMAX)
    image_out = cv2.cvtColor(
        image_out.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    return focus_value, image_out


def combined_focus_measure(image_in):
    gray_image = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)

    # sobel
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_xy = cv2.Sobel(gray_image, cv2.CV_64F, 1, 1, ksize=3)
    combined_gradients = gradient_magnitude + np.abs(sobel_xy)
    sobel_var = np.var(combined_gradients)

    # fswm
    sigma_low = 2.5
    sigma_high = 3.0
    blur_low = cv2.GaussianBlur(gray_image, (0, 0), sigmaX=sigma_low)
    blur_high = cv2.GaussianBlur(gray_image, (0, 0), sigmaX=sigma_high)
    bandpass = blur_low - blur_high
    fswm_var = np.var(bandpass)

    focus_value = sobel_var + 0.5*(fswm_var**0.75)
    normalized_image = cv2.normalize(
        combined_gradients, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image_out = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2RGB)

    return focus_value, image_out


def combined_focus_measure2(image_in):
    gray_image = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)

    # Sobel-based focus value
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_xy = cv2.Sobel(gray_image, cv2.CV_64F, 1, 1, ksize=3)
    combined_gradients = gradient_magnitude + np.abs(sobel_xy)
    sobel_var = np.var(combined_gradients)

    # Compute FFT-based focus value
    window = np.hanning(gray_image.shape[0])[
        :, None] * np.hanning(gray_image.shape[1])[None, :]
    gray_windowed = gray_image * window

    f = np.fft.fft2(gray_windowed)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)

    center_y, center_x = magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2
    low_freq_size = 10
    magnitude_spectrum[center_y - low_freq_size:center_y + low_freq_size,
                       center_x - low_freq_size:center_x + low_freq_size] = 0

    fft_var = np.var(magnitude_spectrum)

    focus_value = sobel_var + (0.5*fft_var/(1e5))

    # magnitude_spectrum_log = 20 * np.log1p(magnitude_spectrum)
    normalized_image = cv2.normalize(
        combined_gradients, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image_out = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2RGB)

    return focus_value, image_out
