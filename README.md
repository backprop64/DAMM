# Detect Any Mouse Model (DAMM) [[project page](https://web.eecs.umich.edu/gkaul/DAMM/)]
- A codebase for single/multi-animal tracking in videos (Kaul et al. 2024).
- Checkout the asssociated [SAM annotation tool](https://github.com/backprop64/sam_annotator) used in this paper
  
## Updates
[Sep 2024] SAM 2 integration for video tracking

[Sep 2024] DAMM accepted into Scientific Reports!

## Setup this codebase locally, tested on a linux system with a GPU 
```bash
# create conda enviornent
conda create -n DAMM python=3.10
conda activate DAMM

#get codebase
git clone https://github.com/backprop64/DAMM 
cd DAMM

# setup SAM 2
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install . 

# setup detectron2, torch, opencv
conda install conda-forge::detectron2
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia 
conda install conda-forge::opencv

# make everything importable
cd - 
python setup.py install 
```
---

## Download Model Weights
```bash
# detect any mouse model/config
wget https://www.dropbox.com/s/39a690qldduxawz/DAMM_weights.pth
wget https://www.dropbox.com/s/wegw8l5zq3vqln0/DAMM_config.yaml


# sam model weights  (models below are ordered from smallest to largest, and you only need 1)
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt #(associated config: sam2_hiera_tiny.yaml)
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt #(associated config: sam2_hiera_small.yaml)
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt #(associated config: sam2_hiera_base_plus.yaml)
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt #(associated config: sam2_hiera_large.yaml)

```

## Track Mice in your Python Scripts
```python
from DAMM.tracking import PromptableVideoTracker

#initilize tracking setup 
mouse_tracker = PromptableVideoTracker(
    sam_config="sam2_hiera_l.yaml", # you dont need to download this, its already in the sam repo
    sam_checkpoint= "path/to/sam2_hiera_large.pt",
    damm_config="path/to/DAMM_config.yaml",
    damm_checkpoint="path/to/DAMM_weights.pth"
)

# Track the first 250 frames of demo_video.mp4
# Save the output and visualization to the output_dir
mouse_tracker.predict_video(
    video_path='demo_video.mp4',
    output_dir='demo_output/',
    batch_size=64,
    start_frame=0,
    end_frame=250,
    visualize=True
)

```
## Track Mice Via the Command line

```bash
conda activate DAMM
cd path/to/DAMM
python track_mice.py \
    --sam_config "path/to/sam2_hiera_l.yaml" \  # Path to the SAM configuration file
    --sam_checkpoint "path/to/sam2_hiera_large.pt" \  # Path to the SAM checkpoint file
    --damm_config "path/to/DAMM_config.yaml" \  # Path to the DAMM configuration file
    --damm_checkpoint "path/to/DAMM_weights.pth" \  # Path to the DAMM weights file
    --video_input "path/to/input/video.mp4" \  # Path to the input video file (required)
    --output_dir "path/to/output/directory/" \  # Directory for output results
    --start_frame 0 \  # Starting frame for processing
    --end_frame 1000 \  # Ending frame for processing
    --visualize true  # Whether to visualize the output (true/false)
```

## Use our system entirely in Google Colab (DAMM+SORT)

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
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1gh3IqFnMd4G2-ao93cgSBOCe2QWQ24FX?usp=sharing) | Manually correcting ID errors | AER Lab |

---



## Citing our work (models and annotation tools)

If our DAMM tool was useful, please cite us!

```
@article{kaul2024damm,
  author    = {Gaurav Kaul and Jonathan McDevitt and Justin Johnson and Ada Eban-Rothschild},
  title = {DAMM for the detection and tracking of multiple animals within complex social and environmental settings},
  journal = {Scientific Reports},
  volume = {14},
  pages = {21366},
  year = {2024},
  doi = {10.1038/s41598-024-72367-2},
  url = {https://doi.org/10.1038/s41598-024-72367-2},
}
```
