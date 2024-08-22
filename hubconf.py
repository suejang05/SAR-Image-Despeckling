# import torch
# from unet import UNet as _UNet

# def unet_carvana(pretrained=False, scale=0.5):
#     """
#     UNet model trained on the Carvana dataset ( https://www.kaggle.com/c/carvana-image-masking-challenge/data ).
#     Set the scale to 0.5 (50%) when predicting.
#     """
#     net = _UNet(n_channels=1, bilinear=True)
#     #if pretrained:
#     #    if scale == 0.5:
#     #        checkpoint = 'https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale0.5_epoch2.pth'
#     #    elif scale == 1.0:
#     #        checkpoint = 'https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale1.0_epoch2.pth'
#     #    else:
#     #        raise RuntimeError('Only 0.5 and 1.0 scales are available')
#     state_dict = torch.hub.load_state_dict_from_url(save_checkpoint, progress=True)
#     if 'mask_values' in state_dict:
#         state_dict.pop('mask_values')
#     net.load_state_dict(state_dict)    

#     return net


import torch
from unet import UNet as _UNet

def unet_custom(pretrained=False, model_path=None):
    """
    UNet model for a custom dataset.
    If pretrained is True, loads the model state from the specified model_path.
    """
    net = _UNet(n_channels=1, bilinear=True)

    if pretrained:
        if model_path is None:
            raise RuntimeError('For pretrained models, model_path must be specified')
        state_dict = torch.load(model_path)
        net.load_state_dict(state_dict)    

    return net

# Example usage:
# net = unet_custom(pretrained=True, model_path='path_to_your_model.pth')

