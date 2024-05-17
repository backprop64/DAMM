import cv2
import numpy as np
import os
import random 
from tqdm import tqdm

def find_video_files(directory):
    video_files = []

    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file has a video extension (you can include other video extensions as needed)
            if file.endswith((".mp4", ".avi", ".mov", ".mkv")):
                video_path = os.path.join(root, file)
                video_files.append(video_path)

    return video_files


import os


def sample_frames(video_paths, k, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    total_sampled = 0
    saved_paths = []

    while total_sampled < k:
        for video_path in video_paths:
            if total_sampled >= k:
                return saved_paths
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            idx = random.randint(0, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                output_path = os.path.join(
                    output_folder, f"{video_name}_frame{idx}.jpg"
                )
                cv2.imwrite(output_path, frame)
                saved_paths.append(output_path)
                total_sampled += 1

            cap.release()

    return saved_paths

def add_padding(image, padding_size):
    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Calculate the new dimensions with padding
    new_height = height + 2 * padding_size
    new_width = width + 2 * padding_size

    # Create a new array with the padded dimensions
    padded_image = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)

    # Copy the original image into the padded array
    padded_image[
        padding_size : new_height - padding_size,
        padding_size : new_width - padding_size,
    ] = image

    return padded_image


def squarify_crop(frame, y1, y2, x1, x2, context=0):
    original_height, original_width = frame.shape[:2]

    # Calculate original bounding box dimensions
    original_width = x2 - x1
    original_height = y2 - y1

    center_x = x1 + (original_width // 2)
    center_y = y1 + (original_height // 2)

    # Find the largest side length
    max_side = max(original_width, original_height)
    context = int(context * max_side)
    padding = max_side + context

    new_x1 = padding + center_x - (max_side // 2) - context
    new_x2 = padding + center_x + (max_side // 2) + context
    new_y1 = padding + center_y - (max_side // 2) - context
    new_y2 = padding + center_y + (max_side // 2) + context

    frame = add_padding(frame, padding)
    resize_image = frame[new_y1:new_y2, new_x1:new_x2]

    return resize_image


def save_cropped_video(frames_list, video_path, output_folder):
    # Sort the frames list by frame number
    output_path = os.path.join(
        output_folder,
        video_path.split(os.sep)[-1][:-4]
        + "_mouse_id_"
        + str(frames_list[-1][-2])
        + "_action_cam.mp4",
    )
    print("saving video to:", output_path)
    # Open the video file
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter object to save the cropped video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (128, 128))

    for frame_info in tqdm(frames_list):
        x1, y1, x2, y2, id, frame_num = frame_info

        # Iterate through frames until the desired frame number
        while video.get(cv2.CAP_PROP_POS_FRAMES) < frame_num:
            _, _ = video.read()

        # Read the frame and crop the region of interest
        ret, frame = video.read()
        if not ret:
            break

        cropped_frame = squarify_crop(frame, y1, y2, x1, x2, context=0.25)

        # Resize the cropped frame to 128x128
        cropped_frame = cv2.resize(cropped_frame, (128, 128))

        # Save the cropped frame to the output video
        out.write(cropped_frame)

    # Release the video capture and writer objects
    video.release()
    out.release()
