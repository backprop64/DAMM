import cv2
import os
from tqdm import tqdm
import numpy as np

def assign_colors(mouse_ids):
    """Assign a unique color to each mouse ID."""
    colors = {}
    for idx, mouse_id in enumerate(mouse_ids):
        colors[mouse_id] = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
    return colors

def visualize_video(frame_data, video_path, output_folder):
    """visualize tracking using the frame data output"""
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Get the list of mouse IDs from the first frame data
    mouse_ids = [key for key in frame_data[0] if key != 'frame_num']
    colors = assign_colors(mouse_ids)
    
    # Prepare to write the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(output_folder, 'output_video.mp4')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each frame
    for frame_info in tqdm(frame_data, desc="Processing frames", unit="frame"):
        frame_num = frame_info["frame_num"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            print(f"Frame {frame_num} could not be read.")
            continue
        
        # Draw polygons
        for mouse_id, coords in frame_info.items():
            if mouse_id == 'frame_num':
                continue

            if len(coords) < 4:
                continue

            # Extract polygon points and scale them to the target space
            points = [(int(coords[i]), int(coords[i+1])) for i in range(0, len(coords), 2)]

            if len(points) < 3:
                continue

            # Draw the polygon
            print('drawing_polygon')
            cv2.polylines(frame, [np.array(points, np.int32)], isClosed=True, color=colors[mouse_id], thickness=2)
        
        # Write the frame to the output video
        out.write(frame)
    
    # Release resources
    cap.release()
    out.release()
    print(f"Video saved to {output_path}")


def mask_to_polygons(mask):
    # converts a binary boolen mask into a polygon of structure [[x1,y1,x2,y2.....xn,yn],...[x1,y1,x2,y2.....xn,yn]]
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        contour = contour.reshape(-1, 2)
        polygon = [int(coord) for point in contour for coord in point]
        polygons.append(polygon)
    return polygons

def save_frame_chunk(video_path, output_folder, start_frame, end_frame):
    # saves a chunk of the output to the correct folder 
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = 0 # sam wants things 0-n for jpg names
    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        file_path = os.path.join(output_folder, f"{frame_idx}.jpg")
        cv2.imwrite(file_path, frame)
        frame_idx += 1
    cap.release()

