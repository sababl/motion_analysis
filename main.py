import cv2
import glob
import numpy as np
from utils import *
from preprocess import *
from connected_components import *

if __name__ == "__main__":
    frames = []
    output_directory = "output"
    bg_img = cv2.imread("data/es1/Background.jpg", cv2.IMREAD_GRAYSCALE)
    # Load frames
    frames = load_image_frames("data/es1/video")

    components = []
    for frame in frames:
        img = cv2.absdiff(frame, bg_img)
        components.append(find_connected_components(img))


    tracks = track_components(components)
    visualize_tracking(frames, components, tracks)
