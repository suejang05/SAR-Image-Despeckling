# import argparse
# import logging
# import os

# import numpy as np
# import torch
# import torch.nn.functional as F
# from PIL import Image
# from torchvision import transforms
# from skimage.metrics import peak_signal_noise_ratio, structural_similarity  # Import metrics

# from unet import UNet
# from utils.utils import plot_img_and_mask

# def predict_img(net, full_img, device, scale_factor=1):
#     net.eval()
#     full_img = full_img.convert("L")
#     img = transforms.ToTensor()(full_img)
#     img = img.unsqueeze(0)
#     img = img.to(device=device, dtype=torch.float32)

#     with torch.no_grad():
#         output = net(img).cpu()
#         output = output.squeeze().numpy()
#         output = (output - output.min()) / (output.max() - output.min())  # Normalize to 0-1

#     return output

# def get_args():
#     parser = argparse.ArgumentParser(description='Predict masks from input images')
#     parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
#                         help='Specify the file in which the model is stored')
#     parser.add_argument('--input-folder', '-i', metavar='INPUT_FOLDER', required=True,
#                         help='Folder of input images')
#     parser.add_argument('--output-folder', '-o', metavar='OUTPUT_FOLDER', required=True,
#                         help='Folder for output masks')
#     parser.add_argument('--gt-folder', '-g', metavar='GT_FOLDER', required=True,
#                         help='Folder of Ground Truth images')
#     parser.add_argument('--viz', '-v', action='store_true',
#                         help='Visualize the images as they are processed')
#     parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
#     parser.add_argument('--scale', '-s', type=float, default=0.5,
#                         help='Scale factor for the input images')
#     parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
#     return parser.parse_args()

# def get_output_filenames(input_folder, output_folder):
#     input_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
#     output_files = [os.path.join(output_folder, f"{os.path.splitext(os.path.basename(f))[0]}_OUT.png") for f in input_files]
#     return input_files, output_files

# def save_image(output, out_filename):
#     output = (output * 255).astype(np.uint8)
#     output_img = Image.fromarray(output)
#     output_img.save(out_filename)

# def compute_metrics(output, gt_img):
#     output = (output * 255).astype(np.uint8)
#     gt_img = np.array(gt_img)

#     psnr_value = peak_signal_noise_ratio(gt_img, output)
#     ssim_value = structural_similarity(gt_img, output, data_range=gt_img.max() - gt_img.min())

#     return psnr_value, ssim_value

# if __name__ == '__main__':
#     args = get_args()
#     logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

#     os.makedirs(args.output_folder, exist_ok=True)
#     in_files, out_files = get_output_filenames(args.input_folder, args.output_folder)

#     net = UNet(n_channels=1, bilinear=args.bilinear)

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logging.info(f'Loading model {args.model}')
#     logging.info(f'Using device {device}')

#     net.to(device=device)
#     state_dict = torch.load(args.model, map_location=device)
#     net.load_state_dict(state_dict)

#     logging.info('Model loaded!')

#     psnr_values = []
#     ssim_values = []
#     top_psnr_images = []
#     top_ssim_images = []

#     for i, filename in enumerate(in_files):
#         logging.info(f'Predicting image {filename} ...')
#         img = Image.open(filename)

#         output = predict_img(net=net,
#                              full_img=img,
#                              device=device,
#                              scale_factor=args.scale
#                              )

#         if not args.no_save:
#             out_filename = out_files[i]
#             save_image(output, out_filename)
#             logging.info(f'Output image saved to {out_filename}')

#         # Load the corresponding Ground Truth image
#         gt_filename = os.path.join(args.gt_folder, os.path.basename(filename))
#         gt_img = Image.open(gt_filename).convert("L")  # Assuming GT is also grayscale

#         # Compute PSNR and SSIM
#         psnr_value, ssim_value = compute_metrics(output, gt_img)
#         psnr_values.append(psnr_value)
#         ssim_values.append(ssim_value)

#         # Store image paths along with PSNR and SSIM for top 10 sorting
#         top_psnr_images.append((psnr_value, filename))
#         top_ssim_images.append((ssim_value, filename))

#         # Keep only the top 10 scores
#         top_psnr_images = sorted(top_psnr_images, reverse=True, key=lambda x: x[0])[:10]
#         top_ssim_images = sorted(top_ssim_images, reverse=True, key=lambda x: x[0])[:10]

#         logging.info(f'PSNR for {filename}: {psnr_value:.4f}')
#         logging.info(f'SSIM for {filename}: {ssim_value:.4f}')

#         if args.viz:
#             logging.info(f'Visualizing results for image {filename}, close to continue...')
#             plot_img_and_mask(img, output)

#     # Sort PSNR and SSIM values
#     sorted_psnr_values = sorted(psnr_values)
#     sorted_ssim_values = sorted(ssim_values)

#     # Calculate the index to exclude the lowest 10%
#     cutoff_index = int(len(sorted_psnr_values) * 0.1)

#     # Exclude the lowest 10% for PSNR and SSIM
#     filtered_psnr_values = sorted_psnr_values[cutoff_index:]
#     filtered_ssim_values = sorted_ssim_values[cutoff_index:]

#     # Print top 10 PSNR and SSIM images with their scores
#     logging.info('Top 10 PSNR images:')
#     for psnr_value, filename in top_psnr_images:
#         logging.info(f'{filename}: PSNR = {psnr_value:.4f}')

#     logging.info('Top 10 SSIM images:')
#     for ssim_value, filename in top_ssim_images:
#         logging.info(f'{filename}: SSIM = {ssim_value:.4f}')

#     # Calculate the average after excluding the lowest 10%
#     average_psnr = np.mean(filtered_psnr_values)
#     average_ssim = np.mean(filtered_ssim_values)

#     logging.info(f'Average PSNR : {average_psnr:.4f}')
#     logging.info(f'Average SSIM : {average_ssim:.4f}')



# import argparse
# import logging
# import os

# import numpy as np
# import torch
# import torch.nn.functional as F
# from PIL import Image
# from torchvision import transforms
# from skimage.metrics import peak_signal_noise_ratio, structural_similarity  # Import metrics

# from unet import UNet 
# from utils.utils import plot_img_and_mask

# def predict_img(net, full_img, device, scale_factor=1):
#     net.eval()
#     # Convert the image to grayscale (1 channel)
#     full_img = full_img.convert("L")
#     img = transforms.ToTensor()(full_img)
#     img = img.unsqueeze(0)
#     img = img.to(device=device, dtype=torch.float32)

#     with torch.no_grad():
#         output = net(img).cpu()
#         output = output.squeeze().numpy()
#         output = (output - output.min()) / (output.max() - output.min())  # Normalize to 0-1

#     return output

# def get_args():
#     parser = argparse.ArgumentParser(description='Predict masks from input images')
#     parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
#                         help='Specify the file in which the model is stored')
#     parser.add_argument('--input-folder', '-i', metavar='INPUT_FOLDER', required=True,
#                         help='Folder of input images')
#     parser.add_argument('--output-folder', '-o', metavar='OUTPUT_FOLDER', required=True,
#                         help='Folder for output masks')
#     parser.add_argument('--gt-folder', '-g', metavar='GT_FOLDER', required=True,
#                         help='Folder of Ground Truth images')  # Added argument for Ground Truth folder
#     parser.add_argument('--viz', '-v', action='store_true',
#                         help='Visualize the images as they are processed')
#     parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
#     parser.add_argument('--scale', '-s', type=float, default=0.5,
#                         help='Scale factor for the input images')
#     parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
#     return parser.parse_args()

# def get_output_filenames(input_folder, output_folder):
#     input_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
#     output_files = [os.path.join(output_folder, f"{os.path.splitext(os.path.basename(f))[0]}_OUT.png") for f in input_files]
#     return input_files, output_files

# def save_image(output, out_filename):
#     output = (output * 255).astype(np.uint8)
#     output_img = Image.fromarray(output)
#     output_img.save(out_filename)

# def compute_metrics(output, gt_img):
#     output = (output * 255).astype(np.uint8)
#     gt_img = np.array(gt_img)

#     psnr_value = peak_signal_noise_ratio(gt_img, output)
#     ssim_value = structural_similarity(gt_img, output, data_range=gt_img.max() - gt_img.min())


#     return psnr_value, ssim_value

# if __name__ == '__main__':
#     args = get_args()
#     logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

#     os.makedirs(args.output_folder, exist_ok=True)
#     in_files, out_files = get_output_filenames(args.input_folder, args.output_folder)

#     net = UNet(n_channels=1, bilinear=args.bilinear)

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logging.info(f'Loading model {args.model}')
#     logging.info(f'Using device {device}')

#     net.to(device=device)
#     state_dict = torch.load(args.model, map_location=device)
#     net.load_state_dict(state_dict)

#     logging.info('Model loaded!')

#     psnr_values = []
#     ssim_values = []

#     for i, filename in enumerate(in_files):
#         logging.info(f'Predicting image {filename} ...')
#         img = Image.open(filename)

#         output = predict_img(net=net,
#                              full_img=img,
#                              device=device,
#                              scale_factor=args.scale
#                              )

#         if not args.no_save:
#             out_filename = out_files[i]
#             save_image(output, out_filename)
#             logging.info(f'Output image saved to {out_filename}')

#         # Load the corresponding Ground Truth image
#         gt_filename = os.path.join(args.gt_folder, os.path.basename(filename))
#         gt_img = Image.open(gt_filename).convert("L")  # Assuming GT is also grayscale

#         # Compute PSNR and SSIM
#         psnr_value, ssim_value = compute_metrics(output, gt_img)
#         psnr_values.append(psnr_value)
#         ssim_values.append(ssim_value)

#         logging.info(f'PSNR for {filename}: {psnr_value:.4f}')
#         logging.info(f'SSIM for {filename}: {ssim_value:.4f}')

#         if args.viz:
#             logging.info(f'Visualizing results for image {filename}, close to continue...')
#             plot_img_and_mask(img, output)

#     # Optionally, you can print the average PSNR and SSIM across all images
#     logging.info(f'Average PSNR: {np.mean(psnr_values):.4f}')
#     logging.info(f'Average SSIM: {np.mean(ssim_values):.4f}')


import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet 
from utils.utils import plot_img_and_mask

def predict_img(net, full_img, device, scale_factor=1):
    net.eval()
    # Convert the image to grayscale (1 channel)
    full_img = full_img.convert("L")
    img = transforms.ToTensor()(full_img)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = output.squeeze().numpy()
        output = (output - output.min()) / (output.max() - output.min())  # Normalize to 0-1

    return output

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input-folder', '-i', metavar='INPUT_FOLDER', required=True,
                        help='Folder of input images')
    parser.add_argument('--output-folder', '-o', metavar='OUTPUT_FOLDER', required=True,
                        help='Folder for output masks')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    return parser.parse_args()
def get_output_filenames(input_folder, output_folder):
    input_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    output_files = [os.path.join(output_folder, f"{os.path.splitext(os.path.basename(f))[0]}_OUT.png") for f in input_files]
    return input_files, output_files

def save_image(output, out_filename):
    output = (output * 255).astype(np.uint8)
    output_img = Image.fromarray(output)
    output_img.save(out_filename)

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    os.makedirs(args.output_folder, exist_ok=True)
    in_files, out_files = get_output_filenames(args.input_folder, args.output_folder)

    net = UNet(n_channels=1, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)

        output = predict_img(net=net,
                             full_img=img,
                             device=device,
                             scale_factor=args.scale
                             )

        if not args.no_save:
            out_filename = out_files[i]
            save_image(output, out_filename)
            logging.info(f'Output image saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, output)