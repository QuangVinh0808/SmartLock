import cv2
import mediapipe as mp
import numpy as np
import tflite_runtime.interpreter as tflite
import subprocess
import os
import time
import serial
from collections import defaultdict

# ===================== CONFIG =====================
MODEL_PATH = "models/facenet.tflite"   # model từ repo pillarpond
USER_DB_DIR = "users"

# Cosine similarity threshold (tune)
COSINE_THRESHOLD = 0.55

REQUIRED_TIME = 5          # giữ mặt đủ N giây mới unlock
SEND_UNLOCK_ONCE = True    # gửi '1' 1 lần khi đủ thời gian, reset về 0 khi mất mặt

# Serial to ESP32
SERIAL_PORT = "/dev/serial0"
SERIAL_BAUD = 9600

# Camera (rpicam-vid, YUV420)
CAM_W, CAM_H = 640, 480
FRAME_SIZE = CAM_W * CAM_H * 3 // 2  # yuv420 = 1.5 bytes/pixel

# ===================== UTIL: READ EXACT BYTES =====================
def read_exact(stream, n: int) -> bytes:
    """
    Pipe stdout read(n) có thể trả thiếu -> frame bị lệch (ghost/đè).
    Hàm này đảm bảo đọc đủ n bytes cho mỗi frame.
    """
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return b""
        buf.extend(chunk)
    return bytes(buf)

# ===================== SERIAL INIT =====================
try:
    ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
    time.sleep(2)
    print("✅ Kết nối Serial tới ESP32 thành công!")
except Exception as e:
    print(f"❌ Lỗi Serial: {e}")
    raise SystemExit(1)

# ===================== FACE DETECTOR (MediaPipe) =====================
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# ===================== TFLITE INIT =====================
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

in_shape = input_details[0]["shape"]       # expected [1,160,160,3]
out_shape = output_details[0]["shape"]     # expected [1,512]
IN_H, IN_W = int(in_shape[1]), int(in_shape[2])
IN_DTYPE = input_details[0]["dtype"]

print(f"✅ Model input: {in_shape}, dtype={IN_DTYPE}")
print(f"✅ Model output: {out_shape}, dtype={output_details[0]['dtype']}")

if IN_DTYPE != np.float32:
    print("⚠️ Model dtype không phải float32. Nếu chạy sai, báo mình để chỉnh preprocess.")

# ===================== PREPROCESS (FaceNet prewhiten) =====================
def prewhiten(img_rgb_float_0_255: np.ndarray) -> np.ndarray:
    x = img_rgb_float_0_255.astype(np.float32)
    mean = np.mean(x)
    std = np.std(x)
    std_adj = max(std, 1.0 / np.sqrt(x.size))
    return (x - mean) / std_adj

def l2_normalize(v: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    return v / (np.linalg.norm(v) + eps)

def get_face_embedding(face_bgr: np.ndarray) -> np.ndarray | None:
    if face_bgr is None or face_bgr.size == 0:
        return None

    face_resized = cv2.resize(face_bgr, (IN_W, IN_H))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB).astype(np.float32)  # 0..255
    face_input = prewhiten(face_rgb)
    face_input = np.expand_dims(face_input, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]["index"], face_input)
    interpreter.invoke()
    emb = interpreter.get_tensor(output_details[0]["index"])[0].astype(np.float32)
    return l2_normalize(emb)

# ===================== DB LOAD =====================
def safe_crop(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    H, W = img.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 0, 3), dtype=img.dtype)
    return img[y1:y2, x1:x2]

def load_user_database() -> dict[str, np.ndarray]:
    if not os.path.exists(USER_DB_DIR):
        print(f"⚠️ Không thấy folder {USER_DB_DIR}/")
        return {}

    buckets = defaultdict(list)

    for filename in os.listdir(USER_DB_DIR):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        path = os.path.join(USER_DB_DIR, filename)
        img = cv2.imread(path)
        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb)
        if not results.detections:
            print(f"⚠️ Không detect được mặt trong {filename}")
            continue

        det = results.detections[0]
        bbox = det.location_data.relative_bounding_box
        H, W = img.shape[:2]
        x = int(bbox.xmin * W)
        y = int(bbox.ymin * H)
        w = int(bbox.width * W)
        h = int(bbox.height * H)

        face_crop = safe_crop(img, x, y, w, h)
        emb = get_face_embedding(face_crop)
        if emb is None:
            continue

        name = os.path.splitext(filename)[0].split("_")[0]
        buckets[name].append(emb)

    db = {}
    for name, embs in buckets.items():
        m = np.mean(np.stack(embs, axis=0), axis=0)
        db[name] = l2_normalize(m)
        print(f"✅ Loaded {name}: {len(embs)} image(s)")

    if not db:
        print("⚠️ DB rỗng. Hãy bỏ ảnh vào users/ (vd: cong_1.jpg, cong_2.jpg)")
    return db

known_faces_db = load_user_database()

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # đã L2 normalize

# ===================== CAMERA PIPE =====================
# NOTE: bỏ --inline để tránh chèn metadata
rpicam_command = [
    "rpicam-vid", "-t", "0",
    "--width", str(CAM_W), "--height", str(CAM_H),
    "--codec", "yuv420",
    "--nopreview",
    "-o", "-"
]
pipe = subprocess.Popen(rpicam_command, stdout=subprocess.PIPE, bufsize=10**8)

# ===================== STATE =====================
face_detected_start_time = None
prev_frame_time = time.time()
prev_state = -1
unlock_sent = False

def send_state(state: int):
    ser.write(b"1" if state == 1 else b"0")
    print(f">>> {'OPEN' if state == 1 else 'CLOSE'}")

try:
    while True:
        raw = read_exact(pipe.stdout, FRAME_SIZE)
        if not raw:
            print("❌ EOF hoặc không đọc được frame từ camera pipe.")
            break

        # yuv420 -> bgr
        yuv = np.frombuffer(raw, dtype=np.uint8).reshape((CAM_H + CAM_H // 2, CAM_W))
        frame_bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detection_results = face_detector.process(rgb_frame)

        any_known_face_present = False
        display_text = "Unknown"
        current_state = 0

        # để vẽ text đúng vị trí khi có bbox
        x = y = fw = fh = 0

        if detection_results.detections and known_faces_db:
            best_name = "Unknown"
            best_score = -1.0
            best_bbox = None

            # chọn 1 bbox tốt nhất (score cao nhất) để hiển thị ổn
            for det in detection_results.detections:
                bbox = det.location_data.relative_bounding_box
                H, W = frame_bgr.shape[:2]
                tx = int(bbox.xmin * W)
                ty = int(bbox.ymin * H)
                tw = int(bbox.width * W)
                th = int(bbox.height * H)

                face_crop = safe_crop(frame_bgr, tx, ty, tw, th)
                emb = get_face_embedding(face_crop)
                if emb is None:
                    continue

                for name, saved_emb in known_faces_db.items():
                    score = cosine_sim(saved_emb, emb)
                    if score > best_score:
                        best_score = score
                        best_name = name
                        best_bbox = (tx, ty, tw, th)

            if best_bbox is not None:
                x, y, fw, fh = best_bbox

            if best_score >= COSINE_THRESHOLD:
                any_known_face_present = True
                display_text = f"{best_name} ({best_score:.2f})"
            else:
                display_text = f"Unknown ({best_score:.2f})" if best_score > -1 else "Unknown"

            color = (0, 255, 0) if any_known_face_present else (0, 0, 255)
            if fw > 0 and fh > 0:
                cv2.rectangle(frame_bgr, (x, y), (x + fw, y + fh), color, 2)
                cv2.putText(frame_bgr, display_text, (x, max(20, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # ======== HOLD-TIME LOGIC ========
        if any_known_face_present:
            if face_detected_start_time is None:
                face_detected_start_time = time.time()
                unlock_sent = False

            elapsed = time.time() - face_detected_start_time
            countdown = max(0.0, REQUIRED_TIME - elapsed)

            if elapsed >= REQUIRED_TIME:
                current_state = 1
                cv2.putText(frame_bgr, "UNLOCKING...", (180, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            else:
                # nếu chưa có bbox thì vẽ ở góc cho khỏi lỗi vị trí
                px = x if fw > 0 else 20
                py = (y + fh + 30) if fh > 0 else 60
                cv2.putText(frame_bgr, f"Hold still: {countdown:.1f}s", (px, py),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            face_detected_start_time = None
            unlock_sent = False
            current_state = 0

        # ======== SERIAL SEND ========
        if SEND_UNLOCK_ONCE:
            if current_state == 1 and not unlock_sent:
                send_state(1)
                unlock_sent = True
                prev_state = 1
            elif current_state == 0 and prev_state != 0:
                send_state(0)
                prev_state = 0
        else:
            if current_state != prev_state:
                send_state(current_state)
                prev_state = current_state

        # ======== FPS ========
        curr_t = time.time()
        dt = curr_t - prev_frame_time
        fps = (1.0 / dt) if dt > 1e-6 else 0.0
        prev_frame_time = curr_t

        cv2.putText(frame_bgr, f"FPS: {int(fps)}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Smart Door (FaceNet) - Fixed", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    try:
        send_state(0)
    except Exception:
        pass
    try:
        pipe.terminate()
    except Exception:
        pass
    cv2.destroyAllWindows()
    try:
        ser.close()
    except Exception:
        pass