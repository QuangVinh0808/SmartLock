import cv2
import face_recognition
import numpy as np
import pickle
import os

from SmartLock.core.detector import HOGFaceDetector

class FaceDatabaseCreator:
    def __init__(self, db_path="face_database.pkl", num_samples=30):
        self.db_path = db_path
        self.num_samples = num_samples
        self.samples = []

    def save_to_pkl(self, name, optimized_vector):
        database = {}
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                database = pickle.load(f)
        
        database[name] = optimized_vector
        
        with open(self.db_path, "wb") as f:
            pickle.dump(database, f)
        print(f"--- Đã lưu thành công database cho: {name} ---")

    def create(self, name, detector_instance):
        cap = cv2.VideoCapture(0)
        self.samples = []
        print(f"Đang thu thập 30 mẫu cho {name}. Hãy di chuyển mặt nhẹ nhàng...")

        while len(self.samples) < self.num_samples:
            ret, frame = cap.read()
            if not ret: break

            boxes = detector_instance.detect(frame)

            if len(boxes) == 1:
                (x1, y1, x2, y2) = boxes[0]
                face_location = [(y1, x2, y2, x1)]
                
                rgb_frame = frame[:, :, ::-1]
                
                encoding = face_recognition.face_encodings(rgb_frame, face_location)
                
                if encoding:
                    self.samples.append(encoding[0])
                    # Vẽ feedback lên màn hình
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"Sample: {len(self.samples)}/30", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.imshow("Create Database", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if len(self.samples) == self.num_samples:
            # Tối ưu: Tính vector trung bình từ 30 mẫu
            optimized_vector = np.mean(self.samples, axis=0)
            self.save_to_pkl(name, optimized_vector)
            return True
        return False


if __name__ == "__main__":
    # Khởi tạo khối Detector của bạn
    my_detector = HOGFaceDetector(detect_every_n_frames=1) # Để 1 cho nhanh khi tạo DB
    
    # Khởi tạo khối tạo DB
    db_creator = FaceDatabaseCreator(num_samples=30)
    
    name_to_register = input("Nhập tên người mới: ")
    db_creator.create(name_to_register, my_detector)