# split_data.py
import os
import shutil
import random

def split_data(source_dir, train_dir, val_dir, split_ratio=0.2):
    """
    Split files from source_dir into train_dir and val_dir based on split_ratio.
    
    Args:
      source_dir (str): Folder containing subfolders "real" and "fake" with extracted frames.
      train_dir (str): Destination folder for training images.
      val_dir (str): Destination folder for validation images.
      split_ratio (float): Fraction of data to assign to validation.
    """
    for category in ["real", "fake"]:
        src = os.path.join(source_dir, category)
        train_category = os.path.join(train_dir, category)
        val_category = os.path.join(val_dir, category)
        os.makedirs(train_category, exist_ok=True)
        os.makedirs(val_category, exist_ok=True)
        files = os.listdir(src)
        random.shuffle(files)
        split_index = int(len(files) * (1 - split_ratio))
        train_files = files[:split_index]
        val_files = files[split_index:]
        for file in train_files:
            shutil.move(os.path.join(src, file), os.path.join(train_category, file))
        for file in val_files:
            shutil.move(os.path.join(src, file), os.path.join(val_category, file))
        print(f"Moved {len(train_files)} images to {train_category}")
        print(f"Moved {len(val_files)} images to {val_category}")

# Define paths
source_data_path = "./data/train"         # Folder where extract_frames.py stored all frames
train_data_path = "./data/train_final"      # Destination for training images
val_data_path = "./data/val"                # Destination for validation images

split_data(source_data_path, train_data_path, val_data_path, split_ratio=0.2)
