import cv2
import os

import numpy as np


def find_connected_components(image, min_size=100):
    """
    Find connected components in a binary image using OpenCV.
    
    Parameters:
    image: Input gray scale image (numpy array)
    min_size: Minimum size of the connected components to consider (int)
    """

    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

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

    return {"filtered_stats":filtered_stats, "filtered_centroids":filtered_centroids}


def track_components(components_list, max_distance=50, min_iou=0.3):
    def iou(box1, box2):
        x1, y1, w1, h1 = box1[:4]
        x2, y2, w2, h2 = box2[:4]
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0

    tracks = []
    for frame_idx in range(len(components_list)-1):
        current_components = components_list[frame_idx]
        next_components = components_list[frame_idx+1]

        for curr_idx, (curr_centroid, curr_box) in enumerate(zip(current_components['filtered_centroids'], 
                                                                current_components['filtered_stats'])):
            for next_idx, (next_centroid, next_box) in enumerate(zip(next_components['filtered_centroids'], 
                                                                     next_components['filtered_stats'])):
                distance = np.linalg.norm(np.array(curr_centroid) - np.array(next_centroid))
                if distance <= max_distance and iou(curr_box, next_box) >= min_iou:
                    tracks.append((frame_idx, curr_idx, frame_idx + 1, next_idx))

    return tracks

def visualize_tracking(frames, components, tracks, output_directory="output"):
    """
    Visualize tracked connected components across frames.

    Parameters:
    frames: List of numpy arrays representing video frames.
    components: List of dictionaries containing 'filtered_stats' and 'filtered_centroids' for each frame.
    tracks: List of tuples tracking components as (frame_idx, component_idx, next_frame_idx, next_component_idx).
    output_directory: Directory to save the output visualization frames.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Assign unique colors for each track
    track_colors = {}

    # Create a mapping from track components to colors
    for track in tracks:
        frame_idx, comp_idx, next_frame_idx, next_component_idx = track

        if (frame_idx, comp_idx) not in track_colors:
            track_colors[(frame_idx, comp_idx)] = tuple(np.random.randint(0, 255, size=3).tolist())
            track_colors[(next_frame_idx, next_component_idx)] = track_colors[(frame_idx, comp_idx)]
        else:
            track_colors[(next_frame_idx, next_component_idx)] = track_colors[(frame_idx, comp_idx)]

    # Convert grayscale frames to colorful by replicating channels
    colorful_frames = [cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if len(frame.shape) == 2 else frame for frame in frames]

    # Draw tracks and bounding boxes
    for frame_idx, frame_components in enumerate(components):
        for comp_idx, (stat, centroid) in enumerate(zip(frame_components['filtered_stats'], frame_components['filtered_centroids'])):
            color = track_colors.get((frame_idx, comp_idx), (255, 255, 255))
            x, y, w, h, _ = stat
            cv2.rectangle(colorful_frames[frame_idx], (x, y), (x + w, y + h), color, 2)
            centroid = tuple(map(int, centroid))
            cv2.circle(colorful_frames[frame_idx], centroid, 5, color, -1)

    # Create a trajectory map on the first frame
    trajectory_map = colorful_frames[-1].copy()
    for track in tracks:
        frame_idx, comp_idx, next_frame_idx, next_comp_idx = track
        curr_centroid = tuple(map(int, components[frame_idx]['filtered_centroids'][comp_idx]))
        next_centroid = tuple(map(int, components[next_frame_idx]['filtered_centroids'][next_comp_idx]))
        color = track_colors[(frame_idx, comp_idx)]
        cv2.line(trajectory_map, curr_centroid, next_centroid, color, 2)

    # Save the trajectory map
    cv2.imwrite(f"{output_directory}/trajectory_map.jpg", trajectory_map)

    # Save the visualized frames
    for i, frame in enumerate(colorful_frames):
        cv2.imwrite(f"{output_directory}/frame_{i:03d}.jpg", frame)

    print(f"Visualized frames and trajectory map saved to {output_directory}")

# Example usage (assuming you have frames, components, and tracks):
# visualize_tracking(frames, components, tracks)
