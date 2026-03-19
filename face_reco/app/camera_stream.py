"""Camera stream utilities."""
import cv2  # pylint: disable=no-member

def start_camera():
    """
    Initialize and return camera capture object.
    
    Returns:
        cv2.VideoCapture: Camera capture
    """
    cap = cv2.VideoCapture(0)  # pylint: disable=no-member
    return cap
