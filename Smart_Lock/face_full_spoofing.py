import cv2
import mediapipe as mp
import numpy as np
import tflite_runtime.interpreter as tflite
import subprocess
import os
import time
import serial
import torch
from src.model_lib.MiniFASNet import MiniFASNetV2   # chỉnh path nếu cần

# ================= CONFIG =================
FACE_MODEL_PATH = "models/mobilefacenet.tflite"
SPOOF_MODEL_PATH = "models/2.7_80x80_MiniFASNetV2.pth"
USER_DB_DIR = "users"

RECOGNITION_THRESHOLD = 0.65
SPOOF_THRESHOLD = 0.8
REQUIRED_TIME = 5

# ================= SERIAL =================
try:
    ser = serial.Serial('/dev/serial0', 9600, timeout=1)
    time.sleep(2)
    print("Serial connected to ESP32!")
except Exception as e:
    print(f"Serial Error: {e}")
    exit()

# ================= FACE DETECTION =================
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

# ================= FACE RECOGNITION MODEL =================
interpreter = tflite.Interpreter(model_path=FACE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
INPUT_IMG_SIZE = input_details[0]['shape'][1:3]

# ================= SPOOF MODEL =================
device = torch.device("cpu")
spoof_model = MiniFASNetV2(conv6_kernel=(5,5)).to(device)
spoof_model.load_state_dict(torch.load(SPOOF_MODEL_PATH, map_location=device))
spoof_model.eval()

# ================= FUNCTIONS =================
def cosine_similarity(a, b):
    return np.dot(a, b)

def get_face_embedding(face_img):
    face_img = cv2.resize(face_img, tuple(INPUT_IMG_SIZE))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = (face_img.astype(np.float32) - 127.5) / 128.0
    face_img = np.expand_dims(face_img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], face_img)
    interpreter.invoke()
    embedding = interpreter.get_tensor(output_details[0]['index'])[0]

    norm = np.linalg.norm(embedding)
    if norm == 0:
        return None
    return embedding / norm

def get_live_score(face_img):
    img = cv2.resize(face_img, (80, 80))
    img = img.transpose(2, 0, 1)
    img = img / 255.0
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = spoof_model(img)
        prob = torch.softmax(output, dim=1)

    return prob[0][1].item()

def load_user_database():
    known_embeddings = {}

    if not os.path.exists(USER_DB_DIR):
        return known_embeddings

    for filename in os.listdir(USER_DB_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(USER_DB_DIR, filename))
            if img is None:
                continue

            results = face_detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if results.detections:
                bbox = results.detections[0].location_data.relative_bounding_box
                h, w, _ = img.shape

                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                fw = int(bbox.width * w)
                fh = int(bbox.height * h)

                face_crop = img[max(0,y):y+fh, max(0,x):x+fw]
                embedding = get_face_embedding(face_crop)

                if embedding is not None:
                    name = os.path.splitext(filename)[0].split('_')[0]
                    known_embeddings[name] = embedding

    return known_embeddings

# ================= LOAD USERS =================
known_faces_db = load_user_database()
print("Loaded users:", list(known_faces_db.keys()))

# ================= CAMERA PIPE =================
rpicam_command = [
    'rpicam-vid',
    '-t', '0',
    '--width', '640',
    '--height', '480',
    '--inline',
    '--codec', 'yuv420',
    '--nopreview',
    '-o', '-'
]

pipe = subprocess.Popen(rpicam_command, stdout=subprocess.PIPE, bufsize=10**8)

# ================= TIMER =================
face_detected_start_time = None
prev_state = -1
prev_frame_time = 0

# ================= MAIN LOOP =================
try:
    while True:
        raw_image_data = pipe.stdout.read(640 * 480 * 3 // 2)
        if len(raw_image_data) == 0:
            break

        yuv_frame = np.frombuffer(raw_image_data, dtype=np.uint8).reshape((720, 640))
        frame_bgr = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        detection_results = face_detector.process(rgb_frame)

        any_known_face_present = False
        display_name = "Unknown"
        similarity_score = 0
        confidence_percent = 0
        live_score = 0
        is_live = False
        current_state = 0

        if detection_results.detections:
            for detection in detection_results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame_bgr.shape

                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                fw = int(bbox.width * w)
                fh = int(bbox.height * h)

                face_crop = frame_bgr[max(0,y):y+fh, max(0,x):x+fw]

                if face_crop.size > 0:

                    # ==== SPOOF CHECK ====
                    live_score = get_live_score(face_crop)
                    is_live = live_score > SPOOF_THRESHOLD

                    # ==== FACE RECOGNITION ====
                    embedding = get_face_embedding(face_crop)

                    if embedding is not None and len(known_faces_db) > 0:
                        best_score = -1
                        best_name = "Unknown"

                        for name, saved in known_faces_db.items():
                            score = cosine_similarity(saved, embedding)
                            if score > best_score:
                                best_score = score
                                best_name = name

                        similarity_score = best_score
                        confidence_percent = max(0, min(100, best_score * 100))

                        if best_score > RECOGNITION_THRESHOLD and is_live:
                            any_known_face_present = True
                            display_name = best_name

                if any_known_face_present and is_live:
                    color = (0, 255, 0)
                elif best_score > RECOGNITION_THRESHOLD and not is_live:
                    color = (0, 255, 255)
                else:
                    color = (0, 0, 255)

                cv2.rectangle(frame_bgr, (x, y), (x + fw, y + fh), color, 2)

                cv2.putText(frame_bgr, display_name,
                            (x, y - 45), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, color, 2)

                cv2.putText(frame_bgr, f"Sim: {similarity_score:.3f}",
                            (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)

                cv2.putText(frame_bgr, f"Conf: {confidence_percent:.1f}%",
                            (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)

                cv2.putText(frame_bgr, f"LiveScore: {live_score:.2f}",
                            (x, y + fh + 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)

        # ================= HOLD TIMER =================
        if any_known_face_present:
            if face_detected_start_time is None:
                face_detected_start_time = time.time()

            elapsed_time = time.time() - face_detected_start_time
            countdown = max(0, REQUIRED_TIME - elapsed_time)

            if elapsed_time >= REQUIRED_TIME:
                current_state = 1
                cv2.putText(frame_bgr, "UNLOCKING...",
                            (200, 240),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 255, 0), 3)
            else:
                cv2.putText(frame_bgr,
                            f"Hold still: {countdown:.1f}s",
                            (20, 460),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)
        else:
            face_detected_start_time = None

        # ================= SERIAL =================
        if current_state != prev_state:
            ser.write(b'1' if current_state == 1 else b'0')
            print(">>>", "OPEN" if current_state == 1 else "CLOSE")
            prev_state = current_state

        # ================= STATUS =================
        status_text = "UNLOCKED" if current_state == 1 else "LOCKED"
        status_color = (0, 255, 0) if current_state == 1 else (0, 0, 255)

        cv2.putText(frame_bgr,
                    f"STATUS: {status_text}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, status_color, 2)

        # ================= FPS =================
        curr_t = time.time()
        fps = 1 / (curr_t - prev_frame_time) if (curr_t - prev_frame_time) > 0 else 0
        prev_frame_time = curr_t

        cv2.putText(frame_bgr,
                    f"FPS: {int(fps)}",
                    (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

        cv2.imshow("Smart Door AI", frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    ser.write(b'0')
    pipe.terminate()
    cv2.destroyAllWindows()
    ser.close()
