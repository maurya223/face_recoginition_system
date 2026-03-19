"""Main face attendance recognition system."""
import os
import time
import logging
import json
from datetime import datetime
import cv2  # pylint: disable=no-member
import numpy as np
from app.detect_face import detect_faces
from app.recognition import get_face_embedding
from app.camera_stream import start_camera
from app import vector_db

LAST_ATTENDANCE_FILE = "last_attendance.json"

# Load last attendance
if os.path.exists(LAST_ATTENDANCE_FILE):
    with open(LAST_ATTENDANCE_FILE, "r", encoding='utf-8') as f:
        last_attendance = json.load(f)
else:
    last_attendance = {}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_INDEX = os.path.join(BASE_DIR, "face_index.faiss")
DB_USERS = os.path.join(BASE_DIR, "users.pkl")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

attendance_log = []
CURRENT_THRESHOLD = 0.8
SHOW_SCORES = False


def main():  # pylint: disable=too-many-locals,too-many-branches,too-many-statements,too-many-nested-blocks
    
    vector_db.load_db()

    # Camera retry logic
    max_retries = 5
    cap = None
    for attempt in range(max_retries):
        cap = start_camera()
        if cap.isOpened():
            break
        logger.warning("Camera init failed (attempt %d/%d)", attempt+1, max_retries)
        time.sleep(1)
    else:
        logger.error("Camera initialization failed after retries")
        return

    print("Face Recognition System Started")
    print("Press 'r' to register new face")
    print("Press ESC to exit")

    # registration state
    register_mode = False
    new_name = ""
    registration_embeddings = []
    registration_frames = 0

    # recognition buffers
    frame_buffers = []
    consec_matches = {}
    last_display_time = {}

    attendance_today = set()

    global CURRENT_THRESHOLD, SHOW_SCORES

    logger.info(
        "Controls: r=register name | t=toggle scores | +/- threshold | ESC=exit"
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame read failed, skipping")
                continue

            faces = detect_faces(frame)

            # Safe buffer append
            if len(frame_buffers) == 0:
                frame_buffers.append([])
            if len(frame_buffers) > 5:
                frame_buffers.pop(0)

            for face_info in faces:
                # FACE COORDINATES
                if isinstance(face_info, tuple):
                    x1, y1, x2, y2 = map(int, face_info)
                else:
                    bbox = face_info.get("bbox", (0, 0, 0, 0))
                    x1, y1, x2, y2 = map(int, bbox)

                face_roi = frame[y1:y2, x1:x2]
                if y2 <= y1 or x2 <= x1 or face_roi.size == 0:
                    continue

                embedding = get_face_embedding(frame, (x1, y1, x2, y2))
                if embedding is None:
                    continue

                embedding = embedding.flatten()
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding /= norm

                frame_buffers[-1].append(embedding)

                # REGISTRATION MODE
                if register_mode and new_name:
                    registration_embeddings.append(embedding)
                    registration_frames += 1

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # pylint: disable=no-member
                    cv2.putText(  # pylint: disable=no-member
                        frame,
                        "Register %s: %d/15" % (new_name, registration_frames),
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,  # pylint: disable=no-member
                        0.6,
                        (0, 255, 255),
                        2,
                    )

                    if registration_frames >= 15:
                        avg_emb = np.mean(registration_embeddings, axis=0)
                        norm = np.linalg.norm(avg_emb)
                        if norm > 0:
                            avg_emb /= norm
                        vector_db.add_user(new_name, avg_emb)
                        logger.info("Registered: %s", new_name)
                        register_mode = False
                        new_name = ""
                        registration_frames = 0
                        registration_embeddings = []
                    continue  # Skip recognition during register

                # RECOGNITION
                recent = [emb for buf in frame_buffers[-3:] for emb in buf]
                if len(recent) == 0:
                    continue

                avg_embedding = np.mean(recent, axis=0)
                name, score = vector_db.search_face(avg_embedding, CURRENT_THRESHOLD)
                face_key = "%d-%d" % (x1, y1)

                # ATTENDANCE LOGIC
                current_time = time.time()
                last_time = last_display_time.get(face_key, 0)
                display_cooldown = 5

                status_text = ""
                color = (0, 255, 0) if name != "unknown" else (0, 165, 255)

                if name != "unknown":
                    if current_time - last_time > display_cooldown:
                        # Log attendance
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        log_entry = "%s,%s,IN" % (timestamp, name)
                        attendance_log.append(log_entry)
                        attendance_today.add(name)
                        last_attendance[face_key] = current_time
                        logger.info("Attendance logged: %s", name)

                # Update cooldown
                last_display_time[face_key] = current_time

                # DRAW RESULT
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # pylint: disable=no-member

                label = name if name != "unknown" else "Unknown"
                if SHOW_SCORES:
                    label += " (%.2f)" % score

                cv2.putText(  # pylint: disable=no-member
                    frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2  # pylint: disable=no-member
                )

                c = consec_matches.get(face_key, 0) + 1
                consec_matches[face_key] = c
                cv2.putText(  # pylint: disable=no-member
                    frame,
                    "Conf:%d" % c,
                    (x2, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,  # pylint: disable=no-member
                    0.5,
                    (255, 255, 0),
                    1,
                )

            # UI INFO
            cv2.putText(  # pylint: disable=no-member
                frame,
                "Threshold: %.2f" % CURRENT_THRESHOLD,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,  # pylint: disable=no-member
                0.6,
                (255, 255, 255),
                1,
            )
            cv2.putText(  # pylint: disable=no-member
                frame,
                "Register: %s" % ('ON' if register_mode else 'OFF'),
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,  # pylint: disable=no-member
                0.6,
                (0, 255, 255),
                1,
            )

            cv2.imshow("Face Attendance System", frame)  # pylint: disable=no-member

            key = cv2.waitKey(1) & 0xFF  # pylint: disable=no-member
            if key == 27:  # ESC
                break
            elif key == ord("r") and not register_mode:
                new_name = input(
                    "Enter name to register (or Enter to cancel): "
                ).strip()
                if new_name:
                    register_mode = True
                    registration_embeddings = []
                    registration_frames = 0
                    logger.info("Registration mode ON for '%s'", new_name)
                else:
                    logger.info("Registration cancelled")
            elif key == ord("t"):
                SHOW_SCORES = not SHOW_SCORES
                logger.info("Scores display: %s", SHOW_SCORES)
            elif key == ord("+"):
                CURRENT_THRESHOLD = min(0.95, CURRENT_THRESHOLD + 0.05)
                logger.info("Threshold: %.2f", CURRENT_THRESHOLD)

    finally:
        # Save last attendance timestamps
        with open(LAST_ATTENDANCE_FILE, "w", encoding='utf-8') as attendance_file:
            json.dump(last_attendance, attendance_file)

        # Save database
        vector_db.save_db()

        with open("attendance.txt", "a", encoding='utf-8') as log_file:
            for log in attendance_log:
                log_file.write("%s\n" % log)

        if not os.path.exists("attendance.csv"):
            with open("attendance.csv", "w", encoding='utf-8') as csv_file:
                csv_file.write("Timestamp,Name,Status\n")

        with open("attendance.csv", "a", encoding='utf-8') as csv_file:
            for log in attendance_log:
                csv_file.write("%s\n" % log)

        if "cap" in locals():
            cap.release()
        cv2.destroyAllWindows()  # pylint: disable=no-member

        print("System closed")


if __name__ == "__main__":
    main()
