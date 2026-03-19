"""Face detection using MTCNN."""
import cv2  # pylint: disable=no-member
import numpy as np
from mtcnn import MTCNN

# load detector once
detector = MTCNN()

def detect_faces(frame):
    """
    Detect faces in frame using MTCNN.
    
    Args:
        frame: BGR image
        
    Returns:
        list: List of (x1, y1, x2, y2) bounding boxes
    """
    if frame is None or frame.size == 0:
        return []

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # pylint: disable=no-member

    try:
        results = detector.detect_faces(rgb_frame)
    except Exception as e:
        print(f"MTCNN Error: {e}")
        return []

    if not results:
        return []

    faces = []
    for res in results:
        x, y, w, h = res["box"]

        # Fix negative values
        x = max(0, x)
        y = max(0, y)

        # Ignore very small faces
        if w < 70 or h < 70:
            continue

        x2 = x + w
        y2 = y + h

        faces.append((x, y, x2, y2))

    return faces
