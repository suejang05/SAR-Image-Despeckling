import cv2 as cv
import numpy as np
from scipy import special
from mpl_toolkits.mplot3d import Axes3D
import os

def gamma_kernel(size, shape, scale):  # shape=alpha, scale=beta
    kernel = np.zeros((size, size), dtype=np.float64)  # filter size
    center = size // 2 
    for x in range(size):
        for y in range(size):
            distance = np.sqrt((x - center)**2 + (y - center)**2)
            kernel[x, y] = (1 / (scale ** shape * special.gamma(shape))) * (distance ** (shape - 1)) * np.exp(-distance / scale)
    kernel /= np.sum(kernel)
    return kernel

def apply_gamma_filter(image, kernel):
    """Apply gamma filter to the image."""
    # Ensure the kernel is 2D
    if len(kernel.shape) != 2:
        raise ValueError("Kernel must be 2D")

    # Apply the filter using OpenCV's filter2D
    filtered_image = cv.filter2D(image, -1, kernel)
    return filtered_image

def process_images_in_folder(input_folder, output_folder, kernel):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):  
            image_path = os.path.join(input_folder, filename)
            image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
            
            if image is None:
                print(f"Failed to load {image_path}")
                continue
            
        
            log_transformed = np.log1p(image.astype(np.float64))
            filtered = apply_gamma_filter(log_transformed, kernel)
            restored_image = np.expm1(filtered)
            restored_image = np.clip(restored_image, 0, 255)
            
            output_path = os.path.join(output_folder, f"{filename}")
            cv.imwrite(output_path, np.uint8(restored_image))
            print(f"Processed and saved: {output_path}")

kernel_size = 3  # must be odd
shape = 2.0 
scale = 2.0  
gamma_filter = gamma_kernel(kernel_size, shape, scale)

input_folder = '/root/workplace/Pytorch-UNet/data/sar2sar'
output_folder = '/root/workplace/Pytorch-UNet/data/sar2sar/pre'

process_images_in_folder(input_folder, output_folder, gamma_filter)