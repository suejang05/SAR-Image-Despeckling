#!/bin/bash

# Set the paths to your local dataset
TRAIN_HQ_LOCAL_PATH="/root/workplace/Pytorch-UNet/datasets/train_hq"
TRAIN_MASKS_LOCAL_PATH="/root/workplace/Pytorch-UNet/datasets/train_masks"

# Ensure the data directories exist
mkdir -p data/imgs
mkdir -p data/masks

# Copy the training images to the data/imgs directory
cp -r $TRAIN_HQ_LOCAL_PATH/* data/imgs/

# Copy the training masks to the data/masks directory
cp -r $TRAIN_MASKS_LOCAL_PATH/* data/masks/

# Rename the masks to match the image filenames
echo "Renaming mask files to match image filenames..."

# Python script to rename mask files
python3 - <<END
import os

train_hq_path = "data/imgs"
train_masks_path = "data/masks"

hq_files = os.listdir(train_hq_path)
mask_files = os.listdir(train_masks_path)

# hq 파일과 동일한 이름으로 변경
for hq_file in hq_files:
    base_name = os.path.splitext(hq_file)[0]
    mask_file_pattern = f"noisy_{base_name}.jpg"
    matching_mask_files = [f for f in mask_files if f == mask_file_pattern]
    
    if matching_mask_files:
        old_mask_file = os.path.join(train_masks_path, matching_mask_files[0])
        new_mask_file = os.path.join(train_masks_path, base_name + "_mask.png")
        os.rename(old_mask_file, new_mask_file)
        print(f"Renamed {old_mask_file} to {new_mask_file}")
    else:
        print(f"No matching mask file found for {hq_file}")

print("All files have been processed.")
END

echo "Data preparation complete."

echo "Listing images in data/imgs:"
ls data/imgs

echo "Listing masks in data/masks:"
ls data/masks
