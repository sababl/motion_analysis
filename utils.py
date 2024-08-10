import cv2
import numpy as np
from matplotlib import pyplot as plt

def binary_map(image):
    _ , bw = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return bw