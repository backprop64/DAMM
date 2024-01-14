from .sort import Sort
import numpy as np
import os
import torch
from tqdm import tqdm
import cv2
import csv
from ..data import add_padding, squarify_crop, save_cropped_video, visualize_tracking
from ..detection import Detector
import warnings
import pandas as pd
import shutil

warnings.filterwarnings("ignore")

import pandas as pd


def read_tracklets(csv_files):
    id_to_bounding_boxes = {}
    id_to_frame_range = {}
    headers = ["frame", "top-x", "top-y", "bottom-x", "bottom-y", "id"]

    for id, file in enumerate(csv_files):
        df = pd.read_csv(file, header=None, names=headers, skiprows=1)

        obj_id = df["id"].astype(int)[0]

        if obj_id not in id_to_bounding_boxes:
            id_to_bounding_boxes[obj_id] = []

        if obj_id not in id_to_frame_range:
            id_to_frame_range[obj_id] = {"min_frame": 0, "max_frame": 0}

        frames = df["frame"].astype(int)
        id_to_bounding_boxes[obj_id] = df[
            ["top-x", "top-y", "bottom-x", "bottom-y"]
        ].to_dict(orient="records")

        id_to_frame_range[obj_id]["min_frame"] = frames.min()
        id_to_frame_range[obj_id]["max_frame"] = frames.max()

    return id_to_bounding_boxes, id_to_frame_range


def merge_keys(id_to_bounding_boxes, id_to_frame_range, key1, key2):
    id_1_end_frame = id_to_frame_range[key1]["max_frame"]
    id_2_start_frame = id_to_frame_range[key2]["min_frame"]

    if id_1_end_frame == id_2_start_frame:
        id_to_bounding_boxes[key1].extend(id_to_bounding_boxes[key2])

    if id_1_end_frame < id_2_start_frame:
        id_to_bounding_boxes[key1].extend(id_to_bounding_boxes[key2])
        missing_frames = id_2_start_frame - id_1_end_frame
        missing_boxes = [
            id_to_bounding_boxes[key1][-1] for box in range(missing_frames)
        ]
        id_to_bounding_boxes[key1].extend(missing_boxes)
        id_to_bounding_boxes[key1].extend(id_to_bounding_boxes[key2])

    if id_1_end_frame > id_2_start_frame:
        id_to_bounding_boxes[key1].extend(id_to_bounding_boxes[key2])
        extra_frames = id_1_end_frame - id_2_start_frame
        id_to_bounding_boxes[key1] = id_to_bounding_boxes[key1][: -1 * extra_frames]
        id_to_bounding_boxes[key1].extend(id_to_bounding_boxes[key2])

    id_to_frame_range[key1]["max_frame"] = id_to_frame_range[key2]["max_frame"]

    # Remove key2 from dictionaries
    del id_to_bounding_boxes[key2]
    del id_to_frame_range[key2]

    return id_to_bounding_boxes, id_to_frame_range


def tracklet_that_ends_first(id_to_frame_range, ids):
    first_end_frame = 10000000
    first_end_id = 0

    for id in ids:
        end = id_to_frame_range[id]["max_frame"]
        if end < first_end_frame:
            first_end_frame = end
            first_end_id = id

    return first_end_id, first_end_frame


def find_next_closest_start_id(id_to_frame_range, start, ids):
    closet_start_frame = 10000000
    closest_start_id = 0

    for id in ids:
        if start < id_to_frame_range[id]["min_frame"] < closet_start_frame:
            closet_start_frame = id_to_frame_range[id]["min_frame"]
            closest_start_id = id

    return closest_start_id, closet_start_frame


def stitch_tracklets(csvs, tracklets_to_maintain=[1, 2]):
    id_to_bounding_boxes, id_to_frame_range = read_tracklets(csvs)

    file2mouse = {}
    for id in tracklets_to_maintain:
        file2mouse[id] = [id]

    for i in range(len(id_to_frame_range.keys())):
        try:
            first_end_id, first_end_frame = tracklet_that_ends_first(
                id_to_frame_range, tracklets_to_maintain
            )

            mergable_tracklets = [
                id for id in id_to_frame_range.keys() if id not in tracklets_to_maintain
            ]

            closest_start_id, closet_start_frame = find_next_closest_start_id(
                id_to_frame_range, first_end_frame, mergable_tracklets
            )

            id_to_bounding_boxes, id_to_frame_range = merge_keys(
                id_to_bounding_boxes, id_to_frame_range, first_end_id, closest_start_id
            )

            print("merged:", first_end_id, closest_start_id)
            file2mouse[first_end_id].append(closest_start_id)

        except:
            print("done merging")

    return id_to_bounding_boxes, id_to_frame_range, file2mouse


class Tracker:
    def __init__(
        self,
        cfg_path: str = None,
        model_path: str = None,
        output_dir=None,
    ):
        self.detector = Detector(
            cfg_path=cfg_path,
            model_path=model_path,
            output_dir=output_dir,
        )
        print("initilizing tracker with weights from: ", model_path)
        self.output_dir = output_dir

    def track_video(
        self,
        video_path,
        num_frames=None,
        threshold=0.7,
        max_age=150,
        min_hits=10,
        iou_threshold=0.1,
        visulize=True,
        num_mice = None
    ):
        mot_tracker = Sort(
            max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold
        )
        detections = self.detector.predict_video(
            video_path=video_path,
            num_frames=num_frames,
            threshold=threshold,
            max_detections=num_mice,
        )
        detections_and_tracks = []
        for i, det in tqdm(enumerate(detections)):
            tracks = mot_tracker.update(det)
            detections_and_tracks.append([i, det.astype(int), tracks.astype(int)])

        video_filename = os.path.basename(video_path).split(".")[0]

        video_output_dir = os.path.join(
            self.output_dir, f"tracking_data_{video_filename}"
        )

        all_csv_paths = self.postprocess_sort_output(
            detections_and_tracks, video_output_dir, num_mice
        )

        if visulize:
            visualize_tracking(
                video_path,
                all_csv_paths,
                os.path.join(video_output_dir, "tracking_visulized.mp4"),
            )

    def postprocess_sort_output(self, detections_and_tracks, video_output_dir,num_mice = None):
        os.makedirs(video_output_dir, exist_ok=True)
        os.makedirs(os.path.join(video_output_dir,'preprocessed_tracks'), exist_ok=True)
        os.makedirs(os.path.join(video_output_dir,'mouse_trajectories'), exist_ok=True)

        max_id = 0
        merged_track_ids = []

        for i, det, tracks in tqdm(detections_and_tracks):
            for track in tracks:
                merged_track_ids.append(track.tolist() + [i])
                id = track.tolist()[-1]
                if id > max_id:
                    max_id = id

        all_trajectories = []
        for id in range(1, max_id + 1):
            trajectory = [t for t in merged_track_ids if t[4] == id]
            if len(trajectory) > 0:
                trajectory = sorted(trajectory, key=lambda x: x[-1])
                all_trajectories.append(trajectory)

        all_csv_paths = []
        for i, data in enumerate(all_trajectories):
            headers = ["top-x", "top-y", "bottom-x", "bottom-y", "id", "frame"]
            tracklet_id = i
            for row in data:
                row[headers.index("id")] = tracklet_id
            # Write the data to a CSV file
            filename = os.path.join(video_output_dir,'preprocessed_tracks', f"tracklet_{str(tracklet_id)}_data.csv")
            all_csv_paths.append(filename)
            print("tracklet id", tracklet_id, "tracking data saved to", filename)
            with open(filename, "w", newline="") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(headers)  # Write the headers as the first row
                csv_writer.writerows(data)

            self.interpolate_tracks(filename)
        
        if num_mice:
            __, __, ids2mouse = stitch_tracklets(
                all_csv_paths, tracklets_to_maintain=[i+1 for i in range(num_mice)]
            )
            print(ids2mouse)
            all_csv_paths = []
            for mouse_id, tracklet_ids in ids2mouse.items():
                print("mouse id", mouse_id, "tracking data saved to", filename)
                for tracklet_num in tracklet_ids:
                    tracklet_filename = os.path.join(video_output_dir,'preprocessed_tracks', f"tracklet_{str(tracklet_num)}_data.csv")
                    mouse_filename = os.path.join(video_output_dir,'mouse_trajectories', f"mouse_{str(mouse_id)}_part_{str(tracklet_num)}_data.csv")
                    shutil.copy(tracklet_filename, mouse_filename)
                    all_csv_paths.append(mouse_filename)
        
        return all_csv_paths

    def interpolate_tracks(self, tracking_data_csv):
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(tracking_data_csv)

        # Find the min and max frame number
        min_frame = df["frame"].min()
        max_frame = df["frame"].max()

        # Create a DataFrame to contain all frames within the range
        all_frames = pd.DataFrame({"frame": range(min_frame, max_frame + 1)})

        # Merge the existing DataFrame with all frames to identify missing frames
        df_all_frames = all_frames.merge(df, on="frame", how="left")

        # Interpolate missing values
        df_all_frames.interpolate(method="linear", inplace=True)

        # Write the updated DataFrame to a new CSV file
        df_all_frames.to_csv(tracking_data_csv, index=False)
