import cv2
from pathlib import Path

def enhance_frame(frame):
    """
    Enhance frame quality for better component detection.
    
    Parameters:
    frame: Input grayscale frame
    
    Returns:
    Enhanced frame
    """
    # Apply contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(frame)
    
    # Apply noise reduction while preserving edges
    enhanced = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
    
    return enhanced

def load_image_frames(directory_path):
    """
    Load a sequence of JPG images from a directory and convert them to grayscale.
    
    Parameters:
    directory_path: String or Path object pointing to the directory containing JPG images
    
    Returns:
    List of numpy arrays representing grayscale frames
    """
    # Convert directory path to Path object for easier handling
    directory = Path(directory_path)
    
    # Get all jpg files in the directory
    image_files = sorted(
        [f for f in directory.glob('*.jpg') if f.is_file()],
        key=lambda x: x.name  # Sort by filename
    )
    
    if not image_files:
        raise ValueError(f"No JPG images found in {directory_path}")
    
    # Load and convert each image
    frames = []
    for img_path in image_files:
        # Read the image
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Warning: Could not read image {img_path}")
            continue
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        
    print(f"Loaded {len(frames)} frames from {directory_path}")
    return frames

