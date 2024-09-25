# Detect Any Mouse Model (DAMM) [[project page](https://web.eecs.umich.edu/gkaul/DAMM/)]
- A codebase for single/multi-animal tracking in videos (Kaul et al. 2024).
- Checkout the asssociated [SAM annotation tool](https://github.com/backprop64/sam_annotator) used in this paper
  
## Updates

*[Sep 2024]* SAM 2 incorperated to automatic mouse tracking 
*[Sep 2024]* DAMM accepted into Scientific Reports

## Setup our codebase locally on a system with a GPU (DAMM+SAM2)

```bash

# create conda enviornent
$ conda create -n sammy6 python=3.10
$ conda activate sammy6

#get codebase
$ git clone https://github.com/backprop64/DAMM 
$ cd DAMM

# setup SAM 2
$ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
$ git clone https://github.com/facebookresearch/segment-anything-2.git
$ cd segment-anything-2
$ pip install . 

# setup detectron2
$ conda install conda-forge::detectron2

# installing detectron2 with conda can potentially revert torch back to a CPU version, so this double checks to ensure we have GPU acesss
$ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia 

# setup remaining packages 
$ conda install conda-forge::opencv

# make DAMM importable
$ cd - 
$ python setup.py install 
```
---


## Using DAMM in your python scripts

```python
from DAMM.tracking import PromptableVideoTracker

sam_config = 'sam2_hiera_l.yaml' # using large sam model
sam_checkpoint = '/nfs/turbo/lsa-adae/kaulg/datasets/DAMM/models/sam2_hiera_large.pt'
damm_config = '/nfs/turbo/lsa-adae/kaulg/datasets/DAMM/models/DAMM_config.yaml'
damm_checkpoint = '/nfs/turbo/lsa-adae/kaulg/datasets/DAMM/models/DAMM_weights.pth'

mouse_tracker = PromptableVideoTracker(sam_config,
                                         sam_checkpoint,
                                         damm_config,
                                         damm_checkpoint)


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
      title     = {DAMM for the detection and tracking of multiple animals within complex social and environmental settings},
      journal   = {bioRxiv},
      year      = {2024}
}
```
