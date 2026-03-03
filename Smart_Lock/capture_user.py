import cv2
import numpy as np
import subprocess
import os
import time

# ================= CONFIG =================
USER_DIR = "users"   # vì bạn đang chạy trong Smart_Lock rồi
WIDTH = 640
HEIGHT = 480

if not os.path.exists(USER_DIR):
    print("Folder 'users' does not exist!")
    exit()

# ================= NHẬP TÊN =================
user_name = input("Enter user name: ").strip()

if user_name == "":
    print("Invalid name!")
    exit()

print("SPACE = Capture | Q = Quit")

# ================= MỞ CAMERA =================
rpicam_command = [
    'rpicam-vid',
    '-t', '0',
    '--width', str(WIDTH),
    '--height', str(HEIGHT),
    '--inline',
    '--codec', 'yuv420',
    '--nopreview',
    '-o', '-'
]

pipe = subprocess.Popen(rpicam_command, stdout=subprocess.PIPE, bufsize=10**8)

img_count = 0

try:
    while True:
        raw_image_data = pipe.stdout.read(WIDTH * HEIGHT * 3 // 2)
        if len(raw_image_data) == 0:
            break

        yuv_frame = np.frombuffer(raw_image_data, dtype=np.uint8).reshape((HEIGHT * 3 // 2, WIDTH))
        frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)

        cv2.putText(frame, f"User: {user_name}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(frame, "SPACE = Capture | Q = Quit", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Capture User", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            timestamp = int(time.time())
            filename = f"{user_name}_{timestamp}.jpg"
            filepath = os.path.join(USER_DIR, filename)

            cv2.imwrite(filepath, frame)
            img_count += 1

            print(f"Saved: users/{filename}")

        elif key == ord('q'):
            break

finally:
    pipe.terminate()
    cv2.destroyAllWindows()

print(f"Captured {img_count} images for {user_name}")
