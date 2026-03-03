import cv2
import mediapipe as mp
import numpy as np
import subprocess
import time

# 1. Khởi tạo MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# 2. Mở luồng bằng lệnh rpicam-vid (cách bạn hay dùng)
# Lệnh này sẽ đẩy dữ liệu hình ảnh thô (raw) vào bộ nhớ đệm
command = [
    'rpicam-vid',
    '-t', '0',                    # Chạy không giới hạn thời gian
    '--width', '640',             # Độ phân giải tối ưu cho Pi 4
    '--height', '480',
    '--inline',
    '--codec', 'yuv420',          # Định dạng nhẹ nhất cho AI
    '--nopreview',                # Không mở cửa sổ phụ của rpicam
    '-o', '-'                     # Đẩy dữ liệu ra output để Python đọc
]

# Chạy lệnh hệ thống
pipe = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10**8)

print("Hệ thống đang đọc từ rpicam... Nhấn Ctrl+C để thoát.")

try:
    while True:
        # Đọc dữ liệu hình ảnh từ rpicam (định dạng YUV420)
        # Kích thước frame cho 640x480 YUV420 là width * height * 1.5
        raw_image = pipe.stdout.read(640 * 480 * 3 // 2)
        
        if len(raw_image) == 0:
            break

        # Chuyển đổi dữ liệu thô sang mảng ảnh OpenCV
        image = np.frombuffer(raw_image, dtype=np.uint8).reshape((720, 640)) # 480 * 1.5 = 720
        frame_bgr = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_I420)

        # Chạy MediaPipe
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        # Vẽ kết quả
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame_bgr, detection)

        cv2.imshow('Face Detection - rpicam backend', frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipe.terminate()
    cv2.destroyAllWindows()