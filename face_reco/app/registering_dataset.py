"""Dataset registration script for face recognition system."""
import os
import logging
import cv2
import numpy as np

from app.detect_face import detect_faces
from app.recognition import get_face_embedding
from app import vector_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_PATH = "dataset/original_images"

vector_db.load_db()

for person in os.listdir(DATASET_PATH):
    if person.startswith("."):
        continue

    person_folder = os.path.join(DATASET_PATH, person)

    if not os.path.isdir(person_folder):
        continue

    logger.info("\nRegistering %s", person)

    embeddings = []

    for img in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img)

        frame = cv2.imread(img_path)  # pylint: disable=no-member

        if frame is None:
            logger.warning("Failed to load: %s", img_path)
            continue

        try:
            faces = detect_faces(frame)
        except (cv2.error, ValueError) as e:
            logger.error("Face detection failed for %s: %s", img_path, e)
            continue

        if len(faces) == 0:
            logger.warning("No faces in %s", img_path)
            continue

        x1, y1, x2, y2 = faces[0]

        if x2 <= x1 or y2 <= y1:
            logger.warning("Invalid bbox in %s", img_path)
            continue

        try:
            embedding = get_face_embedding(frame, (x1, y1, x2, y2))
        except (ValueError, RuntimeError) as e:
            logger.error("Embedding failed for %s: %s", img_path, e)
            continue

        if embedding is None or len(embedding.flatten()) != 512:
            logger.warning("Invalid embedding for %s", img_path)
            continue

        embedding = embedding.flatten()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm

        embeddings.append(embedding)

    if len(embeddings) == 0:
        logger.warning("No valid embeddings for %s", person)
        continue

    avg_embedding = np.mean(embeddings, axis=0)

    vector_db.add_user(person, avg_embedding)

    logger.info("%s registered with %d images", person, len(embeddings))

vector_db.save_db()

logger.info("\nDataset registration completed")
