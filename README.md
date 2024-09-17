# Detect Any Mouse Model (DAMM) [[project page](https://web.eecs.umich.edu/gkaul/DAMM/)]
- A codebase for single/multi-animal tracking in videos (Kaul et al. 2024).
- Checkout the asssociated [SAM annotation tool](https://github.com/backprop64/sam_annotator) used in this paper

## Setup our codebase locally 

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
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1gh3IqFnMd4G2-ao93cgSBOCe2QWQ24FX?usp=sharing) | Manually correcting ID errors | AER Lab |

---


## Using DAMM in your python scripts

```python

from DAMM.tracking import Tracker

   
    sam_checkpoint = 'sam_model.pth'
    sam_model_cfg = 'sam_config.yaml'
        
    damm_checkpoint = 'damm_model.pth'
    damm_model_cfg = 'damm_config.yaml'

    video_path = 'path/to/video'
    output_path = 'path/to/output/folder'

    mouse_tracker = PromptableVideoTracker(checkpoint, model_cfg)
    mouse_tracker.predict_long_video(video_path, output_path, 50)


```
## Using DAMM in the command line
```bash
$ cd DAMM/tracking/
$ conda activate DAMM
$ python promptable_video_tracker.py \
$    --

```
    

## Citing our work (models and annotation tools)

If our DAMM tool was useful, please cite us!

```

@article{kaul2024damm,
      author    = {Gaurav Kaul and Jonathan McDevitt and Justin Johnson and Ada Eban-Rothschild},
      title     = {DAMM for the detection and tracking of multiple animals within complex social and environmental settings},
      journal   = {bioRxiv},
      year      = {2024}
}
```
