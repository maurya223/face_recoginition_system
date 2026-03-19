import os
import sys
import numpy as np
import faiss  # pylint: disable=import-error
import cv2  # pylint: disable=no-member

# Dynamic app path (relative to home/)
APP_PATH = os.path.join(os.path.dirname(__file__), "app")
if APP_PATH not in sys.path:
    sys.path.insert(0, APP_PATH)


def recognize_face(frame):
    """
    Main face recognition function for Django views.
    
    Args:
        frame: BGR image from base64
        
    Returns:
        str: Recognized name or "Unknown"
    """
    if frame is None:
        return "Unknown"

    # Lazy imports from app
    from app.vector_db import load_db, search_face
    from app.detect_face import detect_faces
    from app.recognition import get_face_embedding

    # Load DB
    load_db()

    # Detect faces
    faces = detect_faces(frame)
    if not faces:
        return "Unknown"

    # Pick largest face
    faces = sorted(faces, key=lambda f: (f[2] - f[0]) * (f[3] - f[1]), reverse=True)
    x1, y1, x2, y2 = faces[0]

    # Get embedding
    embedding = get_face_embedding(frame, (x1, y1, x2, y2))
    if embedding is None or len(embedding) != 512:
        return "Unknown"

    # Normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    # Search
    THRESHOLD = 0.85
    name, score = search_face(embedding, threshold=THRESHOLD)
    print("Recognition score: %.3f" % score)

    return name if name != "unknown" else "Unknown"
