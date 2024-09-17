
import cv2
import json
import os
import shutil
import warnings
import argparse
from tqdm import tqdm
import glob 

import torch
import numpy as np

from sam2.build_sam import build_sam2_video_predictor
from ..detection import damm_predictor

#from utils import mask_to_polygons,save_frame_chunk,visualize_video

'''
///////////////////////////
This is a verison of sam2 video prediction that runs on longer videos, 
since the memory used is a function of the frame window size, 
we do long video prediction in batches, where the first frame is 
prompted with an object detector (damm in this case) and the remaining frames are prompted
with the sam2 prediction from the last frame of the previous window
'''

class PromptableVideoTracker:
    def __init__(self, checkpoint, model_cfg):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        print(f"Running SAM 2 Video Segmentation on: {device}")
        self.predictor = build_sam2_video_predictor(
            model_cfg, checkpoint, device=device
        )

    def predict_long_video(self, video_path, output_dir, batch_size, start_frame=0, max_frames=None):
        self.video_path = video_path

        # create folder structure for output
        self.output_dir = output_dir
        self.temp_frames_folder = os.path.join(output_dir, "tmp_video_frames") 
        self.frame_predictions_folder = os.path.join(output_dir, "frame_predictions")  

        self.start_frame = start_frame
        self.max_frames = max_frames
        self.max_frame_number = max_frames - start_frame
        self.batch_size = batch_size
    

        self.total_batches = int(np.ceil(max_frames - start_frame/batch_size))
        self.current_batch = 0 

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_frames_folder, exist_ok=True)
        os.makedirs(self.frame_predictions_folder, exist_ok=True)

        print(' - //////////////////////////////// -')
        print(' - Using sam2 + damm to track video - ')
        print(' - Total Frames:',self.max_frames)
        print(' - Batch Size:',self.batch_size)
        print(' - Number Batches:',self.total_batches)
        print(' - Number Batches:',self.total_batches)
        print(' - Storing predictions in:', self.frame_predictions_folder)

        # get mice prompts using DAMM++ for initilization:
        damm = DAMM(
            "/nfs/turbo/lsa-adae/kaulg/datasets/MOXC/damm_checkpoints/DAMM_config.yaml",
            "/nfs/turbo/lsa-adae/kaulg/datasets/MOXC/damm_checkpoints/DAMM_weights.pth",
        )

        damm.update_detector_settings(threshold=0.85, max_detections=5)
        mice_prompts = damm.get_frame_masks(self.video_path, 0, 3)
        print("damm found", len(mice_prompts), 'mice in the first frame')

        for i in range(self.total_batches):
            mice_prompts = self.predict_chunk(mice_prompts)
            self.current_batch += 1

        # optionally visualize the video, this is kinda slow for long videos 
        # but seperating this step out could enable a batch processing optimization
        # in the visualize_video function
        frame_data = []
        for file in glob.glob(os.path.join(self.frame_predictions_folder,'*.json')):
            frame_data.extend(json.load(open(file))['annotations'])
        sorted_frame_data = sorted(frame_data, key=lambda x: x["frame_num"])
        visualize_video(sorted_frame_data,self.video_path,self.output_dir)
        
        return

    def predict_chunk(self, prompts):

        batch_start_frame_idx = self.start_frame + self.current_batch * self.batch_size
        batch_end_frame_idx = self.start_frame + self.current_batch * self.batch_size + self.batch_size
        batch_end_frame_idx = min(batch_end_frame_idx,self.max_frame_number)
        
        # create batch of frames 
        # idealy we never save frames to a folder and reload, 
        # but this would require modifying sam2 source code
        # tldr its fast enough for our task but it could be even faster

        shutil.rmtree(self.temp_frames_folder)
        os.makedirs(self.temp_frames_folder)  
        save_frame_chunk(self.video_path, self.temp_frames_folder, batch_start_frame_idx, batch_end_frame_idx)

        # initilize inference for sam2
        self.inference_state = self.predictor.init_state(
            video_path=self.temp_frames_folder
        )

        # add intsance level prompts 
        print('prompting with',len(prompts),'mouse prompts' )
        for mouse_prompt in prompts:
            self.prompt_sam(mouse_prompt)

        # track prompted objects with sam2 
        batch_predictions = []
        last_frame_annotations = []
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            frame_prediction = {'frame_num': batch_start_frame_idx + int(out_frame_idx)} 
            # save the last prompt as the np array for the initilization prompt of the next batch
            if out_frame_idx + 1 == self.batch_size:
                for i, out_obj_id in enumerate(out_obj_ids):
                    last_frame_prediction = {}
                    last_frame_prediction['frame_num'] =  0 # this is the prompt for the zero-ith frame of the enxt batch
                    last_frame_prediction['id'] = int(out_obj_id) 
                    last_frame_prediction['mask'] = (out_mask_logits[i] > 0.0).squeeze(0).cpu().numpy()
                    last_frame_annotations.append(last_frame_prediction)

            for i, out_obj_id in enumerate(out_obj_ids):
                frame_prediction[int(out_obj_id)] = mask_to_polygons((out_mask_logits[i] > 0.0).squeeze(0).cpu().numpy())[0]

            batch_predictions.append(frame_prediction)

        batch_data_output_path = os.path.join(self.frame_predictions_folder,str(self.current_batch)+'_'+'sam2_out.json')
        batch_predictions = {'start_frame':batch_start_frame_idx,
                             'end_Frame':batch_end_frame_idx,
                             'annotations':batch_predictions}

        with open(batch_data_output_path, 'w') as f:
            json.dump(batch_predictions, f)

        self.predictor.reset_state(self.inference_state)
        return last_frame_annotations

    def prompt_sam(self, annotation):

        if "mask" in annotation.keys():
            print('added mask prompt for mouse',annotation["id"] )
            self.predictor.add_new_mask(
                inference_state=self.inference_state,
                frame_idx=annotation["frame_num"],
                obj_id=annotation["id"],
                mask=annotation["mask"],
            )

        if "bbox" in annotation.keys():
            print('added bbox prompt for mouse',annotation["id"] )
            box = np.array(annotation["bbox"], dtype=np.float32)
            self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=annotation["frame_num"],
                obj_id=annotation["id"],
                box=box,
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment a video using SAM2VideoSegmenter.")
    
    parser.add_argument('--sam_checkpoint', type=str, required=True, help="Path to the checkpoint file.")
    parser.add_argument('--damm_checkpoint', type=str, required=True, help="Path to the checkpoint file.")
    
    parser.add_argument('--sam_model_cfg', type=str, required=True, help="Path to the model configuration file.")
    parser.add_argument('--damm_model_cfg', type=str, required=True, help="Path to the model configuration file.")
    
    parser.add_argument('--video_path', type=str, required=True, help="Path to the input video file.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to the output directory where results will be saved.")
    
    args = parser.parse_args()
    
    sam_checkpoint = args.sam_checkpoint
    sam_model_cfg = args.sam_model_cfg
        
    damm_checkpoint = args.damm_checkpoint
    damm_model_cfg = args.damm_model_cfg

    video_path = args.video_path
    output_path = args.output_path

    sam2segmenter = PromptableVideoTracker(checkpoint, model_cfg)
    sam2segmenter.predict_long_video(video_path, output_path, 50)
