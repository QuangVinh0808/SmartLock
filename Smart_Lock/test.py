import cv2
import mediapipe as mp
import numpy as np
import tflite_runtime.interpreter as tflite
import subprocess
import os
import time
import serial

# ================= CONFIG =================
MODEL_PATH = "models/mobilefacenet.tflite"
USER_DB_DIR = "users"
RECOGNITION_THRESHOLD = 0.65
REQUIRED_TIME = 5

# ================= SERIAL =================
try:
    ser = serial.Serial('/dev/serial0', 115200, timeout=0.1)
    time.sleep(2)
    print("Serial connected to ESP32!")
except Exception as e:
    print(f"Serial Error: {e}")
    exit()

# Response Time variables
last_unlock_send_time = None
last_response_time = 0

# ================= AI INIT =================
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
INPUT_IMG_SIZE = input_details[0]['shape'][1:3]

# ================= COSINE =================
def cosine_similarity(a, b):
    return np.dot(a, b)

# ================= EMBEDDING =================
def get_face_embedding(face_img):
    if face_img.size == 0:
        return None

    face_img = cv2.resize(face_img, tuple(INPUT_IMG_SIZE))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = (face_img.astype(np.float32) - 127.5) / 128.0
    face_img = np.expand_dims(face_img, axis=0)

    t0 = time.time()
    interpreter.set_tensor(input_details[0]['index'], face_img)
    interpreter.invoke()
    embedding_infer_time = (time.time() - t0) * 1000

    embedding = interpreter.get_tensor(output_details[0]['index'])[0]

    # Normalize embedding
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return None, embedding_infer_time
    return embedding / norm, embedding_infer_time

# ================= LOAD DATABASE =================
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
                result = get_face_embedding(face_crop)
                
                if result[0] is not None:
                    embedding = result[0]
                    name = os.path.splitext(filename)[0].split('_')[0]
                    known_embeddings[name] = embedding

    return known_embeddings

known_faces_db = load_user_database()
print("Loaded users:", list(known_faces_db.keys()))

# ================= CAMERA =================
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
        # Read ACK response from ESP32 to measure response time
        if ser.in_waiting > 0:
            try:
                msg = ser.readline().decode().strip()
                if msg.startswith("ACK") and last_unlock_send_time is not None:
                    last_response_time = (time.time() - last_unlock_send_time) * 1000
                    last_unlock_send_time = None
            except:
                pass

        raw_image_data = pipe.stdout.read(640 * 480 * 3 // 2)
        if len(raw_image_data) == 0:
            break

        yuv_frame = np.frombuffer(raw_image_data, dtype=np.uint8).reshape((720, 640))
        frame_bgr = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        t_detect_start = time.time()
        detection_results = face_detector.process(rgb_frame)
        detection_infer_time = (time.time() - t_detect_start) * 1000

        any_known_face_present = False
        display_name = "Unknown"
        similarity_score = 0
        confidence_percent = 0
        current_state = 0
        embedding_infer_time = 0

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
                    result = get_face_embedding(face_crop)
                    
                    if result[0] is not None and len(known_faces_db) > 0:
                        embedding = result[0]
                        embedding_infer_time = result[1]
                        best_score = -1
                        best_name = "Unknown"

                        for name, saved in known_faces_db.items():
                            score = cosine_similarity(saved, embedding)

                            if score > best_score:
                                best_score = score
                                best_name = name

                        similarity_score = best_score
                        confidence_percent = max(0, min(100, best_score * 100))

                        if best_score > RECOGNITION_THRESHOLD:
                            any_known_face_present = True
                            display_name = best_name

                color = (0, 255, 0) if any_known_face_present else (0, 0, 255)

                cv2.rectangle(frame_bgr, (x, y), (x + fw, y + fh), color, 2)

                cv2.putText(frame_bgr,
                            f"{display_name}",
                            (x, y - 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, color, 2)

                cv2.putText(frame_bgr,
                            f"Sim: {similarity_score:.3f}",
                            (x, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)

                cv2.putText(frame_bgr,
                            f"Conf: {confidence_percent:.1f}%",
                            (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
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
            if current_state == 1:
                last_unlock_send_time = time.time()
                ser.write(b'U\n')
                print(">>>", "OPEN")
            else:
                ser.write(b'0\n')
                print(">>>", "CLOSE")
            prev_state = current_state

        # ================= STATUS DISPLAY =================
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
                    f"FPS: {int(fps)} | Response: {last_response_time:.2f}ms",
                    (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

        # Print inference time to terminal
        if any_known_face_present:
            print(f"[INFER] Detection: {detection_infer_time:.2f}ms | Embedding: {embedding_infer_time:.2f}ms | Total: {detection_infer_time + embedding_infer_time:.2f}ms")

        cv2.imshow("Smart Door AI", frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    ser.write(b'0')
    pipe.terminate()
    cv2.destroyAllWindows()
    ser.close()