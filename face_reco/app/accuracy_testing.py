"""Accuracy testing script for face recognition system."""
import os
import logging
import cv2
import numpy as np

from app.recognition import get_face_embedding
from app import vector_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_PATH = "dataset/original_images"

# Load DB
vector_db.load_db()

correct = 0
total = 0
threshold = 0.8

for person in os.listdir(DATASET_PATH):
    if person.startswith("."):
        continue

    person_folder = os.path.join(DATASET_PATH, person)

    if not os.path.isdir(person_folder):
        continue

    for img in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img)

        frame = cv2.imread(img_path)  # pylint: disable=no-member

        if frame is None:
            logger.warning("Failed to load: %s", img_path)
            continue

        try:
            embedding = get_face_embedding(frame)

            if embedding is None or len(embedding.flatten()) != 512:
                logger.warning("Invalid embedding for %s", img_path)
                continue

            embedding = embedding.flatten()
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding /= norm

            embedding = np.array([embedding]).astype("float32")

            name, score = vector_db.search_face(embedding[0], threshold)

            logger.info("Actual: %s | Predicted: %s (score: %.3f)", person, name, score)

            total += 1

            if name == person:
                correct += 1

        except (ValueError, RuntimeError) as e:
            logger.error("Error processing %s: %s", img_path, e)

accuracy = (correct / total) * 100 if total > 0 else 0

logger.info("\nTotal Images: %d", total)
logger.info("Correct Predictions: %d", correct)
logger.info("Accuracy: %.2f%%", accuracy)
