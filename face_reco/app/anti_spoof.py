"""Anti-spoofing detection using texture, blur, and entropy analysis."""
import cv2  # pylint: disable=no-member
import numpy as np


def anti_spoof_check(frame):
    """
    Detect if frame is live face or spoof (photo/video).
    
    Args:
        frame: BGR image from camera
        
    Returns:
        bool: True if likely live, False if spoof
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # pylint: disable=no-member
    gray = gray.astype("float")

    # Texture variance
    kernel_size = 5
    mean = cv2.blur(gray, (kernel_size, kernel_size))  # pylint: disable=no-member
    sqr_mean = cv2.blur(gray**2, (kernel_size, kernel_size))  # pylint: disable=no-member
    variance = sqr_mean - mean**2

    avg_variance = np.mean(variance)

    if avg_variance < 15 or avg_variance > 200:
        return False

    # Blur detection
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()  # pylint: disable=no-member

    if laplacian_var < 50:
        return False

    # Color entropy
    for i in range(3):
        hist = cv2.calcHist([frame], [i], None, [256], [0, 256])  # pylint: disable=no-member
        hist = hist.flatten() / hist.sum()

        entropy = -np.sum(hist * np.log2(hist + 1e-10))

        if entropy < 3:
            return False

    return True
