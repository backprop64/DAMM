import torch
import os
import argparse
import cv2
import numpy as np
from glob import glob

from ..data.datasets import DetectorDataset
from ..data.visulization import generate_random_pastel_color
from tqdm import tqdm
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.data import build_detection_test_loader, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

pretrained_weights = {
    "coco_detector_50": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
    "coco_detector_101": "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
    "coco_mask_50": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml",
    "coco_mask_101": "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
    "LVIS_mask_50": "LVISv0.5-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml",
    "LVIS_mask_101": "LVISv0.5-InstanceSegmentation/mask_rcnn_R_101_FPN_1x.yaml",
}


class Detector:
    def __init__(
        self,
        cfg_path: str = None,
        model_path: str = None,
        model_type: str = None,
        output_dir="/mouse_detector_output",
    ):
        self.cfg = get_cfg()
        self.update_detector_settings()

        if model_type and not (cfg_path and model_path):
            self.create_new_detector(model_type)
        else:
            self.load_existing_model(cfg_path, model_path)
        

        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print("using device:", self.cfg.MODEL.DEVICE)
        
        self.output_dir = output_dir
        self.cfg.OUTPUT_DIR = output_dir
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)

        self.detector = DefaultPredictor(self.cfg)


    def save_config(self):
        output_file = os.path.join(self.cfg.OUTPUT_DIR, "config.yaml")
        with open(output_file, "w") as f:
            f.write(self.cfg.dump())  # save config to file

    def load_existing_model(self, cfg_path: str = None, model_path: str = None):
        self.cfg.merge_from_file(cfg_path)
        if model_path:
            self.cfg.MODEL.WEIGHTS = model_path
        print("starting model weights coming from:", model_path)
        return

    def create_new_detector(self, model_type: str = None):
        self.cfg.merge_from_file(
            model_zoo.get_config_file(pretrained_weights[model_type])
        )
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            pretrained_weights[model_type]
        )
        print("starting model weights coming from:", self.cfg.MODEL.WEIGHTS)
        return

    def train_detector(self, metadata_files, train_ratio=0.8, test_ratio=0.2):
        self.cfg.OUTPUT_DIR = os.path.join(self.output_dir, "training")

        dataset = DetectorDataset(
            metadata_files,
            self.cfg.OUTPUT_DIR,
        )

        dataset.make_train_test_splits(train_ratio, test_ratio)

        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)

        self.cfg.DATASETS.TRAIN = ("train_split",)
        self.cfg.DATASETS.TEST = ("test_split",)

        # pre fine tune test
        print()
        print("### Per Fine Tuning Evaluation ###")
        print()

        evaluator = COCOEvaluator(
            "test_split", self.cfg, False, output_dir=self.cfg.OUTPUT_DIR
        )

        predictor = DefaultPredictor(self.cfg)
        data_loader = build_detection_test_loader(self.cfg, "test_split")
        coco_metrics = inference_on_dataset(predictor.model, data_loader, evaluator)

        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.SOLVER.IMS_PER_BATCH = 8
        self.cfg.SOLVER.MAX_ITER = 500
        self.cfg.SOLVER.STEPS = ()
        self.cfg.SOLVER.CHECKPOINT_PERIOD = 250
        self.cfg.SOLVER.BASE_LR = 1e-3
        self.cfg.SOLVER.WEIGHT_DECAY = 1e-3
        self.cfg.SOLVER.WARMUP_ITERS = 100

        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

        self.trainer = DefaultTrainer(self.cfg)
        self.save_config()

        self.trainer.resume_or_load(resume=False)
        self.trainer.train()
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")

        # post fine tune test
        print()
        print("### Post Fine Tuning Evaluation ###")
        print()
        
        evaluator = COCOEvaluator(
            "test_split", self.cfg, False, output_dir=self.cfg.OUTPUT_DIR
        )
        predictor = DefaultPredictor(self.cfg)
        data_loader = build_detection_test_loader(self.cfg, "test_split")
        coco_metrics = inference_on_dataset(predictor.model, data_loader, evaluator)

    def predict_img(self, img_paths, output_folder, threshold=0.7, max_detections=2):
        self.update_detector_settings(threshold, max_detections)
        os.makedirs(output_folder, exist_ok=True)
        if isinstance(img_paths, str):
            img_paths = [img_paths]
        visulized_paths = []
        for file_path in tqdm(img_paths):
            img_rgb = cv2.imread(file_path)
            # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            outputs = self.detector(img_rgb)
            instances = outputs["instances"].to("cpu")

            for i in range(len(instances)):
                bbox = instances.pred_boxes.tensor[i].numpy()
                box_color = generate_random_pastel_color()
                cv2.rectangle(
                    img_rgb,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    box_color,
                    2,
                )

            filename = os.path.basename(file_path)
            output_path = os.path.join(output_folder, f"visualized_{filename}")
            visulized_paths.append(output_path)
            cv2.imwrite(output_path, img_rgb)
        return visulized_paths

    def predict_video(
        self,
        video_path,
        num_frames=None,
        threshold=0.7,
        max_detections=2,
    ):
        self.update_detector_settings(threshold, max_detections)
        all_detections = []
        cap = cv2.VideoCapture(video_path)
        max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

        if num_frames:
            num_frames = min(num_frames, max_frames)
        else:
            num_frames = max_frames

        for f in tqdm(range(num_frames),total=len(range(num_frames)),desc="Detecting Mice in Video Frames"):
            ___, frame = cap.read()
            try:
                outputs = self.detector(frame)
                detections = outputs["instances"].to("cpu").pred_boxes.tensor.numpy()
                scores = outputs["instances"].to("cpu").scores.numpy()
                detection_scores = np.concatenate(
                    [detections, np.expand_dims(scores, axis=1)], axis=1
                )
                all_detections.append(detection_scores)
            except:
                all_detections.append(np.empty((0, 5)))

        return all_detections

    def update_detector_settings(self, threshold=0.7, max_detections=2):
        # Set the threshold for detection
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

        # Set the maximum number of detections
        self.cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMAGE = max_detections

        # Update the detector with new settings
        self.detector = DefaultPredictor(self.cfg)
