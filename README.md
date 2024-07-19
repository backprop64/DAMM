# Detect Any Mouse Model (DAMM) [[project page](https://web.eecs.umich.edu/gkaul/DAMM/)]
- A codebase for single/multi-animal tracking in videos (Kaul et al. 2024).
- Checkout the asssociated [SAM annotation tool](https://github.com/backprop64/sam_annotator) used in this paper

## Setup our codebase locally (expects a gpu)

```bash
$ conda create -n DAMM python=3.9 
$ conda activate DAMM
$ git clone https://github.com/backprop64/DAMM 
$ pip install -r DAMM/requirements-gpu.txt
$ python DAMM/setup_gpu.py install 
```
---

## Use our system entirely in Google Colab

### DAMM Tracking Notebook [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AK9Y7PO4HKNRZ05UgmeJB8NyV2it_V0z?usp=sharing)

Use this notebook to track mice in videos. You can either use our default DAMM weights (will be automatically downloaded into the notebook), or use your own weights (created using the fine-tuning notebook; see below).

### DAMM Fine Tuning Notebook [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tVG6HvkxVKCKRzauVEhld3Jp7WZM8QK0?usp=sharing)
Use this notebook to create a dataset, annotate bounding boxes, and fine-tune an object detection model. The fined tuned model can be used for tracking in this notebook, or in the Tracking Notebook.

## Community Contributed Notebooks for Follow-Up Data Analysis of DAMM Tracking Output
| Notebook | Name   | Contributor |
| :---:   | :---: | :---: |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11iYuzp51gdyTJswMUHQONymwqo6feZed?usp=sharing) | Computing Centeroids | AER Lab |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UfktWaedUL5aS4DM8NrYLscKMP_vGwGR?usp=sharing) | Heat map generation | AER Lab |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16S11QrjkpsXIksQf6MqjvfJJLn_fbe-b?usp=sharing) | Kinematics analysis | AER Lab |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19f8eERE5KXh0Sk9RFNPR1JT9FgvXdOY7?usp=sharing) | Annotating experimental setups (e.g., behavioral apparatus) | AER Lab |
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
    "my_dataset/metadata.json"
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
    output_dir="demo_output/tracking_output",
)

damm_tracker.track_video(
    video_path="/path/to/video.mp4",
    threshold=0.7,
    max_detections=2,
    max_age=100,
    min_hits=10,
    iou_threshold=0.1,
    visulize=True,
    num_mice=2,
)
```
