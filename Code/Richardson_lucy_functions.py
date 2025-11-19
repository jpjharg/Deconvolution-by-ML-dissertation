
# Richardson-Lucy functions

import numpy as np
from scipy.signal import convolve2d
import torch
import torch.nn.functional as F




def gaussian_kernel_2d(size, sigma=1.0):
    # Create a 2D gaussian kernel
    ax = torch.linspace(-(size // 2), size // 2, size)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel



def gaussian_kernel_1D(size=21, sigma=2.0):
    # Create a 1D gaussian kernel
    ax = torch.linspace(-(size // 2), size // 2, size)
    kernel = torch.exp(-0.5 * (ax / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel



def poisson_noise(image, scale_factor=1000):
    photon_counts = image * scale_factor
    noisy_photons = torch.poisson(photon_counts)
    noisy_image = noisy_photons / scale_factor
    return torch.clamp(noisy_image, 0, 1)



def convo1d(image, kernel):
    # Ensure image is 3D: [batch, channel, length]
    if image.dim() == 1:
        image = image.unsqueeze(0).unsqueeze(0)  # [L] -> [1, 1, L]
    elif image.dim() == 2:
        image = image.unsqueeze(0)  # [C, L] -> [1, C, L]
    
    # Ensure kernel is 3D: [out_channels, in_channels, kernel_length]
    if kernel.dim() == 1:
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # [L] -> [1, 1, L]
    elif kernel.dim() == 2:
        kernel = kernel.unsqueeze(0)  # [C, L] -> [1, C, L]
    
    padding = (kernel.shape[-1] - 1) // 2
    result = torch.nn.functional.conv1d(image, kernel, padding=padding)
    
    # Return as 1D if input was 1D
    return result.squeeze()



def convo2d(image, kernel):
    if image.dim() == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.dim() == 3:
        image = image.unsqueeze(0)
    
    if kernel.dim() == 2:
        kernel = kernel.unsqueeze(0).unsqueeze(0) 
    elif kernel.dim() == 3:
        kernel = kernel.unsqueeze(0)  
    
    padding = ((kernel.shape[-2] - 1) // 2, (kernel.shape[-1] - 1) // 2)
    
    filtered = F.conv2d(image, kernel, stride=1, padding=padding)
    
    
    return filtered.squeeze(0).squeeze(0)



def richardson_lucy(degr_image, kernel, num_its, estimate=None, c = 1e-10):
    if estimate is None:
        estimate = torch.ones(size=degr_image.shape)
    else:
        estimate = estimate.clone()
    

    if len(kernel.shape) == 1:
        kernel_mirror = torch.flip(kernel, [0])
        conv_func = convo1d
    elif len(kernel.shape) == 2:
        kernel_mirror = torch.flip(kernel, [-1, 0])
        conv_func = convo2d
    else:
        raise ValueError("Kernel must be 1D or 2D tensor")

    for _ in range(num_its):
        blurred_estimate = conv_func(estimate, kernel)
        blurred_estimate = torch.clamp(blurred_estimate, min=c)
        ratio = degr_image / blurred_estimate
        estimate = estimate * conv_func(ratio, kernel_mirror)
        estimate = torch.clamp(estimate, min=0)

    return estimate 



def richardson_lucy_1d(degr_image, kernel, num_its, estimate=None, c=1e-8):
    if estimate is None:
        estimate = torch.ones_like(degr_image)
    else:
        estimate = estimate.clone()

    kernel_mirror = torch.flip(kernel, [-1])

    for _ in range(num_its):
        blurred_estimate = convo1d(estimate, kernel)
        relative_blur = degr_image / (blurred_estimate + c)
        correction = convo1d(relative_blur, kernel_mirror)
        estimate = estimate * correction
        estimate = torch.clamp(estimate, min=0)

    return estimate



def deconvolution(blurred_image, kernel, epsilon=1e-5):
    # Compute Fourier transforms
    P_blurred = torch.fft.fft2(blurred_image)
    B = torch.fft.fft2(kernel.squeeze(), s=blurred_image.shape)
    
    # Avoid division by zero by adding a small constant (epsilon)
    B_conj = torch.conj(B)
    B_magnitude_squared = torch.abs(B)**2
    B_inv = B_conj / (B_magnitude_squared + epsilon)
    
    # Perform deconvolution in the frequency domain
    P_deblurred = P_blurred * B_inv

    # Inverse Fourier transform to get the deblurred image
    deblurred_image = torch.fft.ifft2(P_deblurred).real
    
    return torch.clamp(deblurred_image, 0, 1)
