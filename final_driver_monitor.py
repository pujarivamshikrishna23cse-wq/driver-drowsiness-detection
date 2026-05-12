#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ============================================================
# Driver Drowsiness and Distraction Detection System
# Author: Aditya
# Date: April 2026
# ============================================================

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import csv
import os
from datetime import datetime

# ============================================================
# CONFIGURATION AND THRESHOLDS
# ============================================================

EAR_THRESHOLD = 0.20          # Below this = eyes closing
MAR_THRESHOLD = 0.70          # Above this = yawning
H_LEFT_THRESHOLD = -0.08      # Below this = looking left
H_RIGHT_THRESHOLD = 0.15      # Above this = looking right
V_DOWN_THRESHOLD = 0.15       # Above this = looking down

EAR_CONSEC_FRAMES = 15        # Frames before drowsy alert
MAR_CONSEC_FRAMES = 10        # Frames before yawn alert
DISTRACT_CONSEC_FRAMES = 15   # Frames before distraction alert
PHONE_CONSEC_FRAMES = 5       # Frames before phone alert

PHONE_CLASS_ID = 67           # COCO class index for cell phone

# ============================================================
# LANDMARK INDICES (MediaPipe Face Mesh)
# ============================================================

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH     = [61, 291, 39, 181, 0, 17, 269, 405]

# ============================================================
# LOGGING SETUP
# ============================================================

log_folder = "logs"
os.makedirs(log_folder, exist_ok=True)
log_file = os.path.join(log_folder, "alerts_log.csv")

if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Alert Type"])

def log_alert(alert_type):
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), alert_type])

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_EAR(eye_landmarks):
    """Eye Aspect Ratio — measures how open the eye is"""
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    return (A + B) / (2.0 * C)

def calculate_MAR(mouth_landmarks):
    """Mouth Aspect Ratio — measures how open the mouth is"""
    A = np.linalg.norm(mouth_landmarks[2] - mouth_landmarks[6])
    B = np.linalg.norm(mouth_landmarks[3] - mouth_landmarks[7])
    C = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[1])
    return (A + B) / (2.0 * C)

def get_head_direction(landmarks):
    """Returns horizontal and vertical offset of nose from face center"""
    nose_tip        = landmarks[1]
    left_eye_corner = landmarks[33]
    right_eye_corner= landmarks[263]
    chin            = landmarks[152]
    forehead        = landmarks[10]

    face_width    = np.linalg.norm(right_eye_corner - left_eye_corner)
    face_center_x = (left_eye_corner[0] + right_eye_corner[0]) / 2
    face_center_y = (forehead[1] + chin[1]) / 2

    horizontal_offset = (nose_tip[0] - face_center_x) / face_width

    face_height      = np.linalg.norm(chin - forehead)
    vertical_offset  = (nose_tip[1] - face_center_y) / face_height

    return horizontal_offset, vertical_offset

# ============================================================
# INITIALIZE MODELS
# ============================================================

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

model = YOLO("yolov8n.pt")

# ============================================================
# FRAME COUNTERS AND ALERT FLAGS
# ============================================================

ear_counter      = 0
mar_counter      = 0
distract_counter = 0
phone_counter    = 0

ear_alert_logged      = False
mar_alert_logged      = False
distract_alert_logged = False
phone_alert_logged    = False

# ============================================================
# MAIN LOOP
# ============================================================

cap = cv2.VideoCapture(0)
print("System started. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # --- Phone Detection (YOLOv8) ---
    phone_detected = False
    yolo_results = model(frame, verbose=False)[0]
    for box in yolo_results.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        if cls_id == PHONE_CLASS_ID and conf > 0.4:
            phone_detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"Phone {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    if phone_detected:
        phone_counter += 1
    else:
        phone_counter = 0
        phone_alert_logged = False

    # --- Face Mesh Processing ---
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = np.array([(lm.x * w, lm.y * h)
                                  for lm in face_landmarks.landmark])

            # Calculate metrics
            avg_EAR = (calculate_EAR(landmarks[LEFT_EYE]) +
                       calculate_EAR(landmarks[RIGHT_EYE])) / 2.0
            mar      = calculate_MAR(landmarks[MOUTH])
            h_offset, v_offset = get_head_direction(landmarks)

            # Head direction
            direction  = "Forward"
            distracted = False
            if h_offset < H_LEFT_THRESHOLD:
                direction  = "Looking LEFT"
                distracted = True
            elif h_offset > H_RIGHT_THRESHOLD:
                direction  = "Looking RIGHT"
                distracted = True
            if v_offset > V_DOWN_THRESHOLD:
                direction  = "Looking DOWN"
                distracted = True

            # Update counters
            if avg_EAR < EAR_THRESHOLD:
                ear_counter += 1
            else:
                ear_counter = 0
                ear_alert_logged = False

            if mar > MAR_THRESHOLD:
                mar_counter += 1
            else:
                mar_counter = 0
                mar_alert_logged = False

            if distracted:
                distract_counter += 1
            else:
                distract_counter = 0
                distract_alert_logged = False

            # Display metrics
            cv2.putText(frame, f"EAR: {avg_EAR:.2f}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Dir: {direction}", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # --- Alerts ---
            if ear_counter >= EAR_CONSEC_FRAMES:
                cv2.putText(frame, "DROWSY!", (30, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if not ear_alert_logged:
                    log_alert("DROWSY")
                    ear_alert_logged = True

            if mar_counter >= MAR_CONSEC_FRAMES:
                cv2.putText(frame, "YAWNING!", (30, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if not mar_alert_logged:
                    log_alert("YAWNING")
                    mar_alert_logged = True

            if distract_counter >= DISTRACT_CONSEC_FRAMES:
                cv2.putText(frame, "DISTRACTED!", (30, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if not distract_alert_logged:
                    log_alert("DISTRACTED")
                    distract_alert_logged = True

            if phone_counter >= PHONE_CONSEC_FRAMES:
                cv2.putText(frame, "PHONE DETECTED!", (30, 280),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if not phone_alert_logged:
                    log_alert("PHONE")
                    phone_alert_logged = True

    cv2.imshow("Driver Drowsiness & Distraction Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("System stopped.")


# In[ ]:




