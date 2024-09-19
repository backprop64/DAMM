import glob
import json
import os
import random
import cv2
import numpy as np
import torch
from tqdm import tqdm

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor,MetadataCatalog, DatasetCatalog
from detectron2.data import build_detection_test_loader
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

pretrained_weights = {
    "coco_detector_50": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
    "coco_detector_101": "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
    "coco_mask_50": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml",
    "coco_mask_101": "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
    "LVIS_mask_50": "LVISv0.5-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml",
    "LVIS_mask_101": "LVISv0.5-InstanceSegmentation/mask_rcnn_R_101_FPN_1x.yaml",
}

class DetectorTrainer:
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



class DetectorDataset:
    # seperate class for handling organization and ensambling datasets to enable
    # consitant train, val, and test splits for model training/evaluation
    def __init__(
        self,
        metadata_files,
        output_folder="mouse_detector_output/",
    ):
        if type(metadata_files) == type("string"):
            self.metadata_files = [metadata_files]
        else:
            self.metadata_files = metadata_files

        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        self.combine_datasets()

    def combine_datasets(self, name="combined_metadata"):
        all_datapoints = []
        image_id = 1
        for metadata in self.metadata_files:
            dataset = json.load(open(metadata))["annotations"]
            dataset = sorted(dataset, key=lambda x: x["file_name"])
            for datapoint in dataset:
                datapoint["file_name"] = "/".join(
                    metadata.split(os.sep)[:-2] + [datapoint["file_name"]]
                )
                datapoint["image_id"] = image_id
                image_id += 1
                all_datapoints.append(datapoint)

        merged_dataset = {"annotations": all_datapoints}
        merged_dataset_path = os.path.join(self.output_folder, name + ".json")

        with open(merged_dataset_path, "w") as f:
            json.dump(merged_dataset, f)

        return

    def make_train_test_splits(
        self,
        train_split=0.8,
        test_split=0.2,
    ):
        merged_dataset_path = os.path.join(self.output_folder, "combined_metadata.json")
        dataset = json.load(open(merged_dataset_path))["annotations"]

        dataset = sorted(dataset, key=lambda x: x["file_name"])

        random.shuffle(dataset)

        # create splits
        train_size = int(train_split * len(dataset))
        test_size = int(test_split * len(dataset))

        test_split = dataset[:test_size]
        train_split = dataset[test_size : min(len(dataset), test_size + train_size)]

        # create directory names
        train_split_path = os.path.join(self.output_folder, "train_split.json")
        test_split_path = os.path.join(self.output_folder, "test_split.json")

        # create dataset dict
        test_split_dataset = {"annotations": test_split}
        train_split_dataset = {"annotations": train_split}

        # save datasets
        with open(train_split_path, "w") as f:
            json.dump(train_split_dataset, f)

        with open(test_split_path, "w") as f:
            json.dump(test_split_dataset, f)

        print("##########################")
        print("creating train/test split(s)")
        print("train_size", len(train_split))
        print("test_size", len(test_split))
        print("##########################")

        datasets = glob.glob(os.path.join(self.output_folder, "*.json"))
        for dset in datasets:
            dset_name = dset.split(os.sep)[-1][:-5]
            self.register_dataset(dset, dset_name)

        return

    def register_dataset(self, metadata_path, name):
        DatasetCatalog.register(
            name,
            lambda path=metadata_path: self.get_dataset_dicts(path),
        )
        MetadataCatalog.get(name).set(thing_classes=["tracking target"])
        print("registered dataset:", name)

    def get_dataset_dicts(
        self,
        metadata_path,
    ):
        dataset_annotations = []
        dataset = json.load(open(metadata_path))["annotations"]
        dataset = sorted(dataset, key=lambda x: x["file_name"])

        for datapoint in dataset:
            if not os.path.isfile(datapoint["file_name"]):
                continue
            
            for ann in datapoint["annotations"]:
                ann["bbox_mode"] = BoxMode.XYXY_ABS
                ann["bbox"] = ann["bbox"][0] + ann["bbox"][1]
                ann["category_id"] = 0
                if "segmentation" in ann.keys():
                    ann["segmentation"] = [[int(v) for v in p] for p in ann["segmentation"]]

            dataset_annotations.append(datapoint)

        print("Loaded " + str(len(dataset_annotations)) + " datapoints")
        return dataset_annotations

