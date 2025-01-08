import cv2
import glob
import numpy as np
from utils import *
from preprocess import *

if __name__ == "__main__":
    frames = []
    output_directory = "output"

    # Load frames
    frames = load_image_frames("data/es1/video")
    # Process frames
    components, adjacency_matrices = detect_motion_and_track(frames)
    visualize_and_save_results(frames, components, output_directory)

    # Create and save adjacency map
    adjacency_map_path = os.path.join(output_directory, 'adjacency_map.png')
    create_adjacency_map(adjacency_matrices, components, adjacency_map_path)
