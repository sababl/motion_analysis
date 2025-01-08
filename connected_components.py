import cv2
import numpy as np


def find_connected_components(image, min_size=100):
    """
    Find connected components in a binary image using OpenCV.
    
    Parameters:
    image: Input binary image (numpy array)
    min_size: Minimum size of the connected components to consider (int)
    """

    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

    filtered_stats = []
    filtered_centroids = []
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area >= min_size:  # Only consider components larger than the minimum size
            filtered_stats.append((x, y, w, h, area))
            filtered_centroids.append(centroids[label])
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2) 
            cv2.putText(output_image, str(label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imwrite('connected_components_output.jpg', output_image)

    return labels, filtered_stats, filtered_centroids


org_image = cv2.imread('data/es1/video/frame0260.jpg', cv2.IMREAD_GRAYSCALE)
background = cv2.imread('Background.jpg', cv2.IMREAD_GRAYSCALE)

image = cv2.absdiff(org_image, background)

labels, filtered_stats, centroids = find_connected_components(image)
print(filtered_stats)
print(labels)
print(centroids)