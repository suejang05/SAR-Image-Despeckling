import torch
import torch.nn.functional as F
from tqdm import tqdm

def psnr(img1, img2, max_pixel_value):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    max_pixel_value
    psnr_value = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr_value.item()

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    psnr_total = 0 

    # iterate over the validation set
    with torch.autocast(device_type=device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            images, true_masks = batch['image'], batch['mask']

            # move images and labels to correct device and type
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            true_masks = true_masks.to(device=device, dtype=torch.float32)

            # predict the mask
            masks_pred = net(true_masks)
            max_pixel_value = images.max().item()
            
            psnr_value = psnr(masks_pred, images, max_pixel_value)
            
            psnr_total += psnr_value
    net.train()
    print(max_pixel_value, masks_pred, true_masks)
    return psnr_total / num_val_batches
