import argparse
from DAMM.tracking import PromptableVideoTracker

# Set up argument parser
parser = argparse.ArgumentParser(description='Video Tracking with DAMM and SAM')
parser.add_argument('--sam_config', type=str, default='sam2_hiera_l.yaml', help='Path to SAM configuration file')
parser.add_argument('--sam_checkpoint', type=str, default='DAMM/models/sam2_hiera_large.pt', help='Path to SAM checkpoint file')
parser.add_argument('--damm_config', type=str, default='DAMM/models/DAMM_config.yaml', help='Path to DAMM configuration file')
parser.add_argument('--damm_checkpoint', type=str, default='DAMM/models/DAMM_weights.pth', help='Path to DAMM weights file')
parser.add_argument('--video_input', type=str, required=True, help='Path to input video file')
parser.add_argument('--output_dir', type=str, default='test_out/', help='Directory for output results')
parser.add_argument('--start_frame', type=int, default=0, help='Starting frame for processing')
parser.add_argument('--end_frame', type=int, default=1000, help='Ending frame for processing')
parser.add_argument('--batch_size', type=int, default=100, help='Ending frame for processing')
parser.add_argument('--visualize', type=bool, default=True, help='Whether to visualize the output')

args = parser.parse_args()

mouse_tracker = PromptableVideoTracker(
    args.sam_config,
    args.sam_checkpoint,
    args.damm_config,
    args.damm_checkpoint
)

mouse_tracker.predict_video(
    args.video_input,
    args.output_dir,
    args.batch_size,
    start_frame=args.start_frame,
    end_frame=args.end_frame,
    visualize=args.visualize
)
