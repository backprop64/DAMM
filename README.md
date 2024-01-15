# Detect Any Mouse Model (DAMM)
*A codebase for single/multi-animal tracking in videos (Kaul et al. 2024).

## Setup our Codebase locally
First install the codabase on your computer (expects a GPU)

```bash
$ conda create -n DAMM python=3.9 
$ git clone https://github.com/backprop64/DAMM 
$ pip install -r requirements-gpu.txt
```

## Use our system entirely in Google Colab

### Tracking Notebook [![here](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AK9Y7PO4HKNRZ05UgmeJB8NyV2it_V0z?usp=sharing)

- **Description:**
  - Track mice in your videos using DAMM.
  - Inputs: Video files.
  - Outputs: Tracking data (CSV file with bounding box coordinates per frame) and a visualization video.
  - Specify the maximum number of mice visible in any frame.

---

### Fine-Tune DAMM Notebook 
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tVG6HvkxVKCKRzauVEhld3Jp7WZM8QK0?usp=sharing)
- **Description:**
  - This notebook is designed for three primary functions:
    1. Creating a custom dataset for fine-tuning the DAMM model.
    2. Fine-tuning the DAMM model with the custom dataset.
    3. Tracking animals in your videos using the fine-tuned model.
  - Steps:
    - For dataset creation, use a GUI in Google Colab to sample and annotate images, preparing a dataset suitable for fine-tuning a mouse detector.
    - Fine-tune the DAMM model using this dataset to enhance tracking accuracy in your specific experimental setup.
    - After fine-tuning, use the model to track animals in your videos. This will output tracking data (CSV file with bounding box coordinates per frame) and a visualization video with the tracked animals.
  - Outputs:
    - A newly fine-tuned detector and its configuration file.
    - Tracking data and visualization videos for your animal tracking experiments.

---

### Community Contributed Notebooks for Follow-Up Data Analysis of DAMM Tracking Output
- **Description:**
  - Series of notebooks for analysis on DAMM tracking output data:
    1. Heat map generation.
    2. Kinematics analysis.
    3. Annotating experimental setups (e.g., behavioral apparatus).
--- 

## Example usage (Code API)

```python

from DAMM.detection import Detector
from DAMM.tracking import Tracker
from DAMM.data import sample_frames, find_video_files

CONFIG_PATH = "path/to/config.yaml"
WEIGHTS_PATH = "path/to/model_final.pth"
EXPARAMENT_VIDEOS = "path/to/exparamental_data"

# load DAMM detector
damm_detector = Detector(
    cfg_path=CONFIG_PATH,
    model_path=WEIGHTS_PATH,
    output_dir="demo_output",
)

# find videos within the demo directory
demo_video_paths = find_video_files(directory=EXPARAMENT_VIDEOS)

# sample 50 frames from found videos
sampled_frame_paths = sample_frames(
    demo_video_paths,
    50,
    output_folder="demo_output/sampled_images",
)

# Use default weights to detect mice in images (zero-shot)
damm_detector.predict_img(
    sampled_frame_paths,
    output_folder="demo_output/zero_shot_detection_predictions",
)

# fine tune detector, (hidden step: 100 images were sampled randomly and annotated in collab)
damm_detector.train_detector(
    "/nfs/turbo/justincj-turbo/kaulg/DAMM/demo/data/saline_cno_dataset/metadata.json"
)

# Use fine_tuned weights to detect mice in images (few-shot/50-shot)
damm_detector.predict_img(
    sampled_frame_paths,
    output_folder="demo_output/few_shot_detection_predictions",
)

# Use default weights to initilize a tracker
damm_tracker = Tracker(
    cfg_path="/demo_output/training/config.yaml",
    model_path="demo_output/model_final.pth",
    output_dir="/tracking_output",
)

damm_tracker.track_video(
    video_path="/path/to/video.mp4",
    threshold=0.7,
    max_detections=2,
    max_age=100,
    min_hits=10,
    iou_threshold=0.1,
    visulize=True
)
```
