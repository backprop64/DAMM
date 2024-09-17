import torch
import cv2
import numpy as np 

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


class Detector:
    def __init__(self, cfg_path: str = None, model_path: str = None):
        self.cfg = get_cfg()
        self.cfg.set_new_allowed(True)  # Add this line before merging the file :) fixes key not found error
        self.cfg.merge_from_file(cfg_path)
        self.cfg.MODEL.WEIGHTS = model_path
        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.detector = DefaultPredictor(self.cfg)

    def update_detector_settings(self, threshold=0.7, max_detections=2):
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        self.cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMAGE = max_detections
        self.detector = DefaultPredictor(self.cfg)

    def get_image_masks(self, image_path: str, top_n: int = 5):
        # Load the image from file
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image file {image_path}.")

        # Perform detection
        outputs = self.detector(image)
        instances = outputs["instances"].to("cpu")

        # Extract masks, bounding boxes, and confidence scores
        masks = instances.pred_masks.numpy()
        scores = instances.scores.numpy()
        boxes = instances.pred_boxes.tensor.numpy()  # Get bounding boxes

        # Get indices of top_n masks based on confidence scores
        top_indices = np.argsort(scores)[::-1][:top_n]

        # Create a list of detection annotations
        top_detections = []
        for idx in top_indices:
            mask_dict = {
                "id": idx,
                "mask": masks[idx].tolist(),  # Convert mask to list for easy serialization
                "confidence": float(scores[idx]),
                "bbox": boxes[idx].tolist()  # Add bounding box
            }
            top_detections.append(mask_dict)
        return top_detections

    def get_frame_masks(self, video_path: str, frame_number: int, top_n: int = 5):
        # Open the video file and get to the specific frame
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Could not read frame number {frame_number} from video.")

        # detect mice
        outputs = self.detector(frame)
        instances = outputs["instances"].to("cpu")

        # Extract masks, bounding boxes, and confidence scores
        masks = instances.pred_masks.numpy()
        scores = instances.scores.numpy()
        boxes = instances.pred_boxes.tensor.numpy()  # Get bounding boxes

        # Get indices of top_n masks based on confidence scores
        top_indices = np.argsort(scores)[::-1][:top_n]

        # create a list of detection annotations
        top_detections = []
        for idx in top_indices:
            mask_dict = {
                "frame_num": frame_number,
                "id": idx,
                "mask": masks[idx],
                "confidence": float(scores[idx]),
                "bbox": boxes[idx].tolist()  # Add bounding box
            }
            top_detections.append(mask_dict)
        return top_detections

    def update_detector_settings(self, threshold=0.7, max_detections=2):
        # Set the threshold for detection
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

        # Set the maximum number of detections
        self.cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMAGE = max_detections

        # Update the detector with new settings
        self.detector = DefaultPredictor(self.cfg)
