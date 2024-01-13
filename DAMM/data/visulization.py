import os
import cv2
import torch
from tqdm import tqdm
import numpy as np
import random
import csv
from collections import defaultdict


def generate_random_pastel_color():
    # Hardcoded range for pastel colors
    min_value = 150
    max_value = 255
    max_variation = 30

    r = random.randint(min_value, max_value)
    g = random.randint(min_value, max_value)
    b = random.randint(min_value, max_value)

    # Apply variations to the color components
    r = max(min(r + random.randint(-max_variation, max_variation), 255), 0)
    g = max(min(g + random.randint(-max_variation, max_variation), 255), 0)
    b = max(min(b + random.randint(-max_variation, max_variation), 255), 0)

    return (r, g, b)

def visualize_detections():
    return 0

def visualize_tracking(video_path, csv_files, output_path):

    print("visualizing frames")
    
    frame_data_dict = {}
    id_color_map = {}

    for i in range(1, len(csv_files) + 1):
        id_color_map[i] = generate_random_pastel_color()

    all_frames = []
    for file in csv_files:
        with open(file, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                obj_id = int(float(file.split('_')[-4]))
                frame_num = int(float(row["frame"]))
                top_x = int(float(row["top-x"]))
                top_y = int(float(row["top-y"]))
                bottom_x = int(float(row["bottom-x"]))
                bottom_y = int(float(row["bottom-y"]))
                all_frames.append(frame_num)
                if frame_num not in frame_data_dict:
                    frame_data_dict[frame_num] = []
                frame_data_dict[frame_num].append(
                    (obj_id, (top_x, top_y, bottom_x, bottom_y))
                )

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    max_frame_num = max(all_frames)

    for frame_num in tqdm(range(max_frame_num)):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num in frame_data_dict:
            for obj_data in frame_data_dict[frame_num]:
                obj_id, box_coords = obj_data
                top_x, top_y, bottom_x, bottom_y = box_coords
                cv2.rectangle(
                    frame, (top_x, top_y), (bottom_x, bottom_y), id_color_map[obj_id], 2
                )
                cv2.putText(
                    frame,
                    f"ID: {obj_id}",
                    (top_x, top_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    id_color_map[obj_id],
                    2,
                )

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
