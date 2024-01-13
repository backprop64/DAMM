# Detect Any Mouse Model (DAMM)
*A codebase for single/multi-animal tracking in videos (Kaul et al. 2024).

## Overview
- Use our robust out-of-box default DAMM model with impressive zero-shot transfer abilities, or enhance its performance by fine-tuning with additional training examples.
  - **Zero-Shot Tracking with DAMM**: Utilize the `DAMM_tracking_out_of_box` notebook. The system automatically employs the default settings (model weights and config files) included with the pretrained model.
  - **Fine-Tuning DAMM**: Upload your fine-tuned model and configuration files. Use the `Fine_Tune_DAMM` notebook for dataset annotation, DAMM fine-tuning, and animal tracking in videos using your fine-tuned model.

## Notebooks

### Tracking Notebook [![here](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qiTIqScLwH7kfp_o5Z1t7UBHNykMptE9?usp=sharing)

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
