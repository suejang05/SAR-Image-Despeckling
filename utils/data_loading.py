import torchvision.transforms as transforms
import os
import logging
import numpy as np
import torch
from PIL import Image
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.distributions as dist
from torch.distributions import Gamma
import torch.nn as nn

def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)



# ########## Gamma distribution filering part ##########
# def gamma_kernel(size: int, shape: float, scale: float) -> torch.Tensor:
#     gamma_dist = torch.distributions.Gamma(shape, scale)
#     kernel = gamma_dist.sample((size, size))
#     kernel = kernel / torch.sum(kernel)
#     return kernel
# kernel = gamma_kernel(size, shape, scale)

# class GammaDenoiseFilter(torch.nn.Module):
#     def __init__(self, kernel_size: int, shape:float, scale:float):
#         super(GammaDenoiseFilter, self).__init__()
#         self.kernel_size = kernel_size
#         self.shape = shape
#         self.scale = scale
#         self.kernel = gamma_kernel(kernel_size, shape, scale)
        
#     def forward(self, x):
#         # Apply Gamma filter
#         x = F.conv2d(x, self.kernel, padding=self.kernel_size // 2, groups=3)
#         return x
    
# denoise_filter = GammaDenoiseFilter(kernel_size=5, shape=2.0, scale=2.0)
# # Assuming 'image' is a batch of images with shape [batch_size, channels, height, width]
# image = torch.randn(1, 1, 256, 256)  # Batch size of 1, 1 color channels, 256*256 image

# denoised_image = denoise_filter(image)
# ########## Gamma distribution filering part ##########



class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, transform=None):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.transform = transforms.Compose([ 
        transforms.ToTensor()
        ])
        # List image and mask files
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        
        # Initialize unique mask values if needed
        self.mask_values = []  # Placeholder, adjust if necessary

    def __len__(self):
        return len(self.ids)


    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        
        mask = load_image(mask_file[0]).convert("L")
        img = load_image(img_file[0]).convert("L")

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        # # Apply preprocessing
        # img = self.preprocess(img, scale=1)
        # mask = self.preprocess(mask, scale=1)

        # Convert to PIL images for transforms
        img = np.array(img)
        mask = np.array(mask)

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        
        return {
            'image': torch.as_tensor(np.array(img)).float().contiguous(),
            'mask': torch.as_tensor(np.array(mask)).float().contiguous()
        }
