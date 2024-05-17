import glob
import json
import os
import random

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

class DetectorDataset:
    # seperate class for handling organization and datasets to enable
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

