import cv2
import mediapipe as mp
import numpy as np
import tflite_runtime.interpreter as tflite
import subprocess
import os
import time
import serial

# --- CẤU HÌNH CỦA BẠN ---
MODEL_PATH = "models/mobilefacenet.tflite"
USER_DB_DIR = "users"
RECOGNITION_THRESHOLD = 1.0
REQUIRED_TIME = 5  # Số giây cần duy trì nhận diện để mở cửa

# --- CẤU HÌNH SERIAL ---
try:
    ser = serial.Serial('/dev/serial0', 9600, timeout=1)
    time.sleep(2)
    print("Kết nối Serial tới ESP32 thành công!")
except Exception as e:
    print(f"Lỗi Serial: {e}")
    exit()

# --- KHỞI TẠO AI ---
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
INPUT_IMG_SIZE = input_details[0]['shape'][1:3]

def get_face_embedding(face_img):
    if face_img.size == 0: return None
    face_img = cv2.resize(face_img, tuple(INPUT_IMG_SIZE))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = (face_img.astype(np.float32) - 127.5) / 128.0
    face_img = np.expand_dims(face_img, axis=0)
    interpreter.set_tensor(input_details[0]['index'], face_img)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

def load_user_database():
    known_embeddings = {}
    if not os.path.exists(USER_DB_DIR): return known_embeddings
    for filename in os.listdir(USER_DB_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(USER_DB_DIR, filename))
            if img is None: continue
            results = face_detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if results.detections:
                bbox = results.detections[0].location_data.relative_bounding_box
                h, w, _ = img.shape
                face_crop = img[int(bbox.ymin*h):int((bbox.ymin+bbox.height)*h), int(bbox.xmin*w):int((bbox.xmin+bbox.width)*w)]
                embedding = get_face_embedding(face_crop)
                if embedding is not None:
                    name = os.path.splitext(filename)[0].split('_')[0]
                    known_embeddings[name] = embedding
    return known_embeddings

known_faces_db = load_user_database()

# --- CAMERA ---
rpicam_command = ['rpicam-vid', '-t', '0', '--width', '640', '--height', '480', '--inline', '--codec', 'yuv420', '--nopreview', '-o', '-']
pipe = subprocess.Popen(rpicam_command, stdout=subprocess.PIPE, bufsize=10**8)

# --- BIẾN ĐIỀU KHIỂN THỜI GIAN ---
face_detected_start_time = None  # Thời điểm bắt đầu nhận diện thấy người quen
prev_state = -1
prev_frame_time = 0

try:
    while True:
        raw_image_data = pipe.stdout.read(640 * 480 * 3 // 2)
        if len(raw_image_data) == 0: break

        yuv_frame = np.frombuffer(raw_image_data, dtype=np.uint8).reshape((720, 640))
        frame_bgr = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detection_results = face_detector.process(rgb_frame)

        any_known_face_present = False
        display_name = "Unknown"
        current_state = 0 # 0 là đóng, 1 là mở

        if detection_results.detections:
            for detection in detection_results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame_bgr.shape
                x, y, fw, fh = int(bbox.xmin*w), int(bbox.ymin*h), int(bbox.width*w), int(bbox.height*h)
                face_crop = frame_bgr[max(0,y):y+fh, max(0,x):x+fw]

                if face_crop.size > 0:
                    embedding = get_face_embedding(face_crop)
                    if embedding is not None:
                        min_dist = min([np.linalg.norm(saved - embedding) for saved in known_faces_db.values()] + [float("inf")])
                        
                        if min_dist < RECOGNITION_THRESHOLD:
                            any_known_face_present = True
                            # Tìm tên người đó để hiển thị
                            for name, saved in known_faces_db.items():
                                if np.linalg.norm(saved - embedding) == min_dist:
                                    display_name = name
                        
                        color = (0, 255, 0) if any_known_face_present else (0, 0, 255)
                        cv2.rectangle(frame_bgr, (x, y), (x + fw, y + fh), color, 2)
                        cv2.putText(frame_bgr, f"{display_name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # --- LOGIC ĐẾM 5 GIÂY ---
        if any_known_face_present:
            if face_detected_start_time is None:
                face_detected_start_time = time.time() # Bắt đầu bấm giờ
            
            elapsed_time = time.time() - face_detected_start_time
            countdown = max(0, REQUIRED_TIME - elapsed_time)
            
            if elapsed_time >= REQUIRED_TIME:
                current_state = 1 # Đã đủ 5s -> Cho phép mở cửa
                cv2.putText(frame_bgr, "UNLOCKING...", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            else:
                # Hiển thị đồng hồ đếm ngược trên màn hình
                cv2.putText(frame_bgr, f"Hold still: {countdown:.1f}s", (x, y + fh + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            face_detected_start_time = None # Reset nếu không thấy người quen hoặc người rời đi

        # --- GỬI SERIAL ---
        if current_state != prev_state:
            ser.write(b'1' if current_state == 1 else b'0')
            print(f">>> {'OPEN' if current_state == 1 else 'CLOSE'}")
            prev_state = current_state

        # Hiển thị FPS
        curr_t = time.time()
        fps = 1/(curr_t-prev_frame_time) if (curr_t-prev_frame_time) > 0 else 0
        prev_frame_time = curr_t
        cv2.putText(frame_bgr, f"FPS: {int(fps)}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Smart Door", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    ser.write(b'0')
    pipe.terminate()
    cv2.destroyAllWindows()
    ser.close()