"""Face embedding extraction using FaceNet."""
import cv2  # pylint: disable=no-member
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load FaceNet models
mtcnn = MTCNN(
    image_size=160,
    margin=0,
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=True,
    device=device,
)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

def get_face_embedding(frame, face_bbox=None):
    """
    Extract 512-dim FaceNet embedding from frame/bbox or auto-detect.
    
    Args:
        frame: BGR image
        face_bbox: (x1,y1,x2,y2) tuple or None for auto-detect
        
    Returns:
        np.array: 512-dim normalized embedding or None
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # pylint: disable=no-member

    if face_bbox is None:
        boxes, probs = mtcnn.detect(rgb_frame)
        if boxes is None or len(boxes) == 0:
            return np.zeros(512)
        best_idx = np.argmax(probs)
        box = boxes[best_idx]
        face_bbox = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))

    x1, y1, x2, y2 = map(int, face_bbox)
    face_img = rgb_frame[y1:y2, x1:x2]
    if face_img.size == 0:
        return np.zeros(512)

    pil_img = Image.fromarray(face_img)

    aligned = mtcnn(pil_img)
    if aligned is None:
        return None

    embedding = resnet(aligned.unsqueeze(0))
    embedding = embedding.detach().cpu().numpy().flatten()

    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding /= norm

    return embedding
