#!/usr/bin/env python3
"""
Embed dataset images directly into Django SQLite db.sqlite3.
Processes dataset/Original_Images/*/, computes avg embedding per person,
saves to home.User.face_embedding as pickled numpy array.
"""

import os
import sys
import django
import cv2
import numpy as np
import logging
import pickle

# Setup Django (standalone script)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'face_reco'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'face_reco.settings')
django.setup()

from home.models import User
from app.detect_face import detect_faces
from app.recognition import get_face_embedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dataset_path = "../dataset/Original_Images"  # Relative to face_reco/

processed = 0
skipped = 0

for person in os.listdir(dataset_path):
    if person.startswith('.'):
        continue
        
    person_folder = os.path.join(dataset_path, person)
    if not os.path.isdir(person_folder):
        continue

    logger.info(f"\nProcessing {person}...")

    embeddings = []

    for img_file in os.listdir(person_folder):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        img_path = os.path.join(person_folder, img_file)
        frame = cv2.imread(img_path)
        if frame is None:
            logger.warning(f"Failed to load: {img_path}")
            skipped += 1
            continue

        try:
            faces = detect_faces(frame)
        except Exception as e:
            logger.warning(f"Detection failed {img_path}: {e}")
            skipped += 1
            continue

        if not faces:
            logger.warning(f"No faces in {img_path}")
            skipped += 1
            continue

        x1, y1, x2, y2 = faces[0]  # Take first face
        if x2 <= x1 or y2 <= y1:
            skipped += 1
            continue

        try:
            embedding = get_face_embedding(frame, (x1, y1, x2, y2))
        except Exception as e:
            logger.warning(f"Embedding failed {img_path}: {e}")
            skipped += 1
            continue

        if embedding is None or len(embedding.flatten()) != 512:
            logger.warning(f"Invalid embedding {img_path}")
            skipped += 1
            continue

        embedding = embedding.flatten()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm
        embeddings.append(embedding)

    if not embeddings:
        logger.warning(f"No valid embeddings for {person}")
        continue

    avg_embedding = np.mean(embeddings, axis=0)
    embedding_pickled = pickle.dumps(avg_embedding)

    # Save to SQLite
    user, created = User.objects.update_or_create(
        name=person,
        defaults={'face_embedding': embedding_pickled}
    )
    
    if created:
        logger.info(f"CREATED {person} ({len(embeddings)} images)")
    else:
        logger.info(f"UPDATED {person} ({len(embeddings)} images)")
    
    processed += 1

logger.info(f"\nCompleted: {processed} people embedded into db.sqlite3, {skipped} images skipped")
logger.info("Verify: cd face_reco && python manage.py shell 'from home.models import User; print(len(User.objects.all()))'")

