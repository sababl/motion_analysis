import cv2
import glob
import os
import numpy as np
import argparse
import matplotlib as plt
from utils import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Detecting motion of objects in a scene")

    # Add arguments
    parser.add_argument("-d", "--data",  type=str, help="path to the data folder", default="./data/es1/video")
    # parser.add_argument("--optional_arg", type=str, help="An optional argument", default="default_value")

    # Parse the arguments
    args = parser.parse_args()

    image_paths = sorted(glob.glob(os.path.join(args.data, '*.*')))

    # Filter out only supported image files if necessary
    image_paths = [path for path in image_paths if path.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm'))]

    # Load the images
    images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
    mask = binary_map(images[0])
    print(mask.shape)   

    plt.figure(figsize=(7, 3.5))
    plt.subplot(1, 2, 1)
    plt.imshow(images[0], cmap='gray')
    plt.axis("off")
    plt.title("original image")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.axis("off")
    plt.title("mask")
    plt.subplots_adjust(wspace=0.05, left=0.01, bottom=0.01, right=0.99, top=0.9)

    plt.show()