import cv2
import mediapipe as mp
import numpy as np
import tflite_runtime.interpreter as tflite
import subprocess
import os
import time

# --- CẤU HÌNH CỦA BẠN ---
MODEL_PATH = "models/mobilefacenet.tflite" # Đảm bảo đúng đường dẫn tới file model
USER_DB_DIR = "users"                      # Thư mục chứa ảnh mẫu của bạn
RECOGNITION_THRESHOLD = 1.0                # Ngưỡng nhận diện (càng nhỏ càng khắt khe, 0.6-1.0 là ổn cho MobileFaceNet)
                                           # Nếu số khoảng cách < THRESHOLD -> Nhận diện thành công

# --- KHỞI TẠO AI VÀ HỆ THỐNG ---

# 1. Khởi tạo MediaPipe Face Detection (Dùng để tìm khuôn mặt)
# Model_selection=0 cho khoảng cách gần, min_detection_confidence=0.5 là đủ
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# 2. Khởi tạo MobileFaceNet TFLite (Dùng để nhận diện)
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Lấy kích thước input mà MobileFaceNet yêu cầu (thường là 112x112)
INPUT_IMG_SIZE = input_details[0]['shape'][1:3] 
print(f"MobileFaceNet yêu cầu ảnh đầu vào kích thước: {INPUT_IMG_SIZE[0]}x{INPUT_IMG_SIZE[1]}")

# --- CÁC HÀM XỬ LÝ ---

def get_face_embedding(face_img):
    """
    Trích xuất "dấu vân tay" khuôn mặt (embedding) từ ảnh đã cắt.
    Ảnh đầu vào phải là ảnh khuôn mặt đã được cắt và định dạng BGR.
    """
    if face_img.size == 0:
        return None

    # Chuẩn hóa ảnh về đúng chuẩn của MobileFaceNet
    face_img = cv2.resize(face_img, tuple(INPUT_IMG_SIZE))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB) # Chuyển BGR sang RGB
    face_img = (face_img.astype(np.float32) - 127.5) / 128.0 # Chuẩn hóa về [-1, 1]
    face_img = np.expand_dims(face_img, axis=0) # Thêm chiều batch (1, H, W, C)
    
    interpreter.set_tensor(input_details[0]['index'], face_img)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0] # Trả về embedding 1xN

def load_user_database():
    """
    Nạp dữ liệu khuôn mặt mẫu từ thư mục USER_DB_DIR.
    Mỗi file ảnh sẽ được xử lý để tạo ra một embedding và lưu vào database.
    """
    known_embeddings = {}
    print(f"\n--- Đang nạp dữ liệu người dùng từ '{USER_DB_DIR}' ---")
    if not os.path.exists(USER_DB_DIR):
        print(f"Thư mục '{USER_DB_DIR}' không tồn tại. Hãy tạo và thêm ảnh mẫu!")
        return known_embeddings

    for filename in os.listdir(USER_DB_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(USER_DB_DIR, filename)
            img = cv2.imread(filepath)
            if img is None:
                print(f"Không thể đọc file ảnh: {filepath}")
                continue

            # Dùng MediaPipe để tìm khuôn mặt trong ảnh mẫu
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_detector.process(rgb_img)

            if results.detections:
                # Lấy khuôn mặt đầu tiên được tìm thấy trong ảnh mẫu
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = img.shape
                x, y, fw, fh = int(bbox.xmin * w), int(bbox.ymin * h), \
                               int(bbox.width * w), int(bbox.height * h)
                
                # Cắt lấy khuôn mặt
                face_crop = img[max(0, y):y + fh, max(0, x):x + fw]
                if face_crop.size > 0:
                    embedding = get_face_embedding(face_crop)
                    if embedding is not None:
                        # Lấy tên từ tên file (ví dụ: 'Cong_1.jpg' -> 'Cong')
                        name = os.path.splitext(filename)[0].split('_')[0]
                        known_embeddings[name] = embedding
                        print(f"    - Đã nạp: '{name}' từ '{filename}'")
                else:
                    print(f"    - Không thể cắt khuôn mặt từ '{filename}'")
            else:
                print(f"    - Không tìm thấy khuôn mặt trong ảnh mẫu: '{filename}'")
    
    print(f"--- Đã nạp thành công {len(known_embeddings)} khuôn mặt vào database ---\n")
    return known_embeddings

# --- NẠP DATABASE NGƯỜI DÙNG KHI KHỞI ĐỘNG ---
known_faces_db = load_user_database()

# --- KHỞI TẠO CAMERA RASPBERRY PI (DÙNG rpicam-vid) ---
# Lệnh này đã hoạt động ổn định trên máy bạn
rpicam_command = [
    'rpicam-vid', '-t', '0', '--width', '640', '--height', '480',
    '--inline', '--codec', 'yuv420', '--nopreview', '-o', '-'
]
pipe = subprocess.Popen(rpicam_command, stdout=subprocess.PIPE, bufsize=10**8)

print("Hệ thống Smart Door đã khởi động. Nhấn 'q' tại cửa sổ để thoát.")

# --- VÒNG LẶP CHÍNH CỦA HỆ THỐNG ---
prev_frame_time = 0

try:
    while True:
        # 1. Đọc frame từ rpicam-vid
        raw_image_data = pipe.stdout.read(640 * 480 * 3 // 2)
        if len(raw_image_data) == 0:
            print("Không nhận được dữ liệu từ camera. Đang thoát.")
            break

        # Chuyển đổi dữ liệu YUV420 từ rpicam sang định dạng BGR của OpenCV
        yuv_frame = np.frombuffer(raw_image_data, dtype=np.uint8).reshape((720, 640)) # Chiều cao 480 + 1/2 cho YUV
        frame_bgr = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)

        # 2. Phát hiện khuôn mặt bằng MediaPipe
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detection_results = face_detector.process(rgb_frame)

        display_name = "Unknown"
        display_distance = float("inf")
        bbox_coords = None

        if detection_results.detections:
            for detection in detection_results.detections:
                # Lấy bounding box của khuôn mặt
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame_bgr.shape
                x, y, fw, fh = int(bbox.xmin * w), int(bbox.ymin * h), \
                               int(bbox.width * w), int(bbox.height * h)
                
                # Đảm bảo bounding box hợp lệ
                x, y = max(0, x), max(0, y)
                fw, fh = min(w - x, fw), min(h - y, fh)
                
                face_crop = frame_bgr[y:y + fh, x:x + fw]

                if face_crop.size > 0:
                    current_face_embedding = get_face_embedding(face_crop)
                    
                    if current_face_embedding is not None:
                        # 3. So sánh với database người dùng
                        min_distance = float("inf")
                        recognized_name = "Unknown"

                        for name, saved_embedding in known_faces_db.items():
                            distance = np.linalg.norm(saved_embedding - current_face_embedding) # Tính khoảng cách Euclidean
                            if distance < min_distance:
                                min_distance = distance
                                recognized_name = name
                        
                        display_name = recognized_name if min_distance < RECOGNITION_THRESHOLD else "Unknown"
                        display_distance = min_distance
                        bbox_coords = (x, y, fw, fh)

                        # Nếu nhận diện thành công, có thể thêm logic mở cửa ở đây
                        if display_name != "Unknown":
                            print(f"ACCESS GRANTED for {display_name} (Distance: {display_distance:.2f})")
                            # --- LOGIC MỞ CỬA CỦA BẠN (GPIO) SẼ Ở ĐÂY ---
                            # Ví dụ: active_gpio_pin_to_open_door()
                        else:
                            print(f"ACCESS DENIED (Unknown Person, Distance: {display_distance:.2f})")

        # 4. Hiển thị kết quả lên màn hình
        if bbox_coords:
            x, y, fw, fh = bbox_coords
            color = (0, 255, 0) if display_name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame_bgr, (x, y), (x + fw, y + fh), color, 2)
            cv2.putText(frame_bgr, f"{display_name} ({display_distance:.2f})", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else: # Nếu không tìm thấy mặt nào
            cv2.putText(frame_bgr, "Looking for a face...", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


        # Tính và hiển thị FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_frame_time)
        prev_frame_time = current_time
        cv2.putText(frame_bgr, f"FPS: {int(fps)}", (20, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


        cv2.imshow("Smart Door - Face Recognition", frame_bgr)

        # Thoát khi nhấn 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Đã xảy ra lỗi nghiêm trọng: {e}")

finally:
    print("\nĐóng hệ thống Smart Door.")
    pipe.terminate() # Đảm bảo tắt tiến trình rpicam-vid
    cv2.destroyAllWindows()