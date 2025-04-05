# extract_frames.py
import os
import cv2
import json

# Parameters for frame extraction
frame_interval = 30  # Extract every 30th frame

# Define destination directories for extracted frames
output_train_real = './data/train/real'
output_train_fake = './data/train/fake'

os.makedirs(output_train_real, exist_ok=True)
os.makedirs(output_train_fake, exist_ok=True)

def extract_frames_from_video(video_path, output_folder, frame_interval=30):
    """
    Extract frames from a video file and save them as JPEG images.
    
    Args:
      video_path (str): Path to the video file.
      output_folder (str): Folder where extracted frames will be saved.
      frame_interval (int): Save one frame every 'frame_interval' frames.
      
    Returns:
      int: Number of frames saved.
    """
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            # Construct filename: videoName_frame<frame_number>.jpg
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            frame_filename = os.path.join(output_folder, f"{base_name}_frame{count}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frames += 1
        count += 1
    cap.release()
    return saved_frames

def process_dataset(dataset_path, metadata_file):
    """
    Process a dataset folder: read metadata and extract frames from each video.
    
    Args:
      dataset_path (str): Path to the dataset folder (e.g., './dataset0').
      metadata_file (str): Name of the metadata JSON file (e.g., 'metadata.json').
    """
    metadata_path = os.path.join(dataset_path, metadata_file)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Iterate over each video file and its associated label
    for video_file, label in metadata.items():
        # If label is a dict, extract actual label using key 'label' (adjust key if needed)
        if isinstance(label, dict):
            label_str = label.get('label', 'fake')
        else:
            label_str = label

        video_path = os.path.join(dataset_path, video_file)
        if not os.path.exists(video_path):
            print(f"Video file {video_file} not found in {dataset_path}.")
            continue
        
        # Determine destination folder based on label
        if label_str.lower() == 'real':
            output_folder = output_train_real
        else:
            output_folder = output_train_fake
        
        frames_saved = extract_frames_from_video(video_path, output_folder, frame_interval)
        print(f"Extracted {frames_saved} frames from {video_file} as {label_str}.")

# Process both dataset0 and dataset1
print("Processing dataset0...")
process_dataset('./dataset0', 'metadata.json')

print("Processing dataset1...")
process_dataset('./dataset1', 'metadata.json')

print("Frame extraction and merging complete!")
