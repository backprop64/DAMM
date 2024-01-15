# Detect Any Mouse Model (DAMM)
*A codebase for single/multi-animal tracking in videos (Kaul et al. 2024).

## Setup our Codebase locally
First install the codabase on your computer (expects a GPU)

```bash
$ conda create -n DAMM python=3.9 
$ git clone https://github.com/backprop64/DAMM 
$ pip install -r requirements-gpu.txt
```
---

## Use our system entirely in Google Colab

### Tracking Notebook [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AK9Y7PO4HKNRZ05UgmeJB8NyV2it_V0z?usp=sharing)
- Use this notebook to track mice in videos using default DAMM weights (will be downloaded in the notebook), or using your own weights (created using the fine tuning notebook; see below). This notebook only optionally uses custom weights/config file. 

### Fine Tuning DAMM Notebook [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tVG6HvkxVKCKRzauVEhld3Jp7WZM8QK0?usp=sharing)
- Use this notebook to create a dataset, annotate bouning boxes, and fine tune a object detection model that will be used in the Tracking Notebook for tracking. All you need to use this notebok is a video or directory containing videos
---

## Community Contributed Notebooks for Follow-Up Data Analysis of DAMM Tracking Output
| Notebook | Name   | Contributor |
| :---:   | :---: | :---: |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/backprop64/DAMM) | Heat map generation | AER Lab |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/backprop64/DAMM) | Kinematics analysis | AER Lab |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/backprop64/DAMM) | Annotating experimental setups (e.g., behavioral apparatus) | AER Lab |
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
