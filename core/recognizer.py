import face_recognition
import numpy as np
import pickle
import os

class FaceRecognizer:
    def __init__(self, db_path="face_database.pkl", tolerance=0.4):
        self.db_path = db_path
        self.tolerance = tolerance
        self.known_faces = self.load_database()

    def load_database(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                return pickle.load(f)
        return {}

    def save_database(self):
        with open(self.db_path, "wb") as f:
            pickle.dump(self.known_faces, f)

    def update_adaptive(self, name, current_encoding, alpha=0.1):
        """Cập nhật vector theo cơ chế Adaptive Learning"""
        old_encoding = self.known_faces[name]
        # Công thức trộn vector để thích nghi với thay đổi ngoại hình
        updated_encoding = (1 - alpha) * old_encoding + alpha * current_encoding
        self.known_faces[name] = updated_encoding
        self.save_database()

    def identify(self, frame, boxes):
        """
        Nhận diện dựa trên danh sách boxes từ HOGFaceDetector
        boxes format: [(x1, y1, x2, y2), ...]
        """
        face_locations = [(y1, x2, y2, x1) for (x1, y1, x2, y2) in boxes]
        rgb_frame = frame[:, :, ::-1] 
        
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        face_names = []
        for i, face_encoding in enumerate(face_encodings):
            name = "Stranger"
            
            if self.known_faces:
                known_names = list(self.known_faces.keys())
                known_vectors = list(self.known_faces.values())
                
                # So khớp
                matches = face_recognition.compare_faces(known_vectors, face_encoding, tolerance=self.tolerance)
                face_distances = face_recognition.face_distance(known_vectors, face_encoding)
                
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_names[best_match_index]
                        # Tự động cập nhật vector khi nhận diện đúng (Adaptive)
                        self.update_adaptive(name, face_encoding)
            
            face_names.append(name)
            
        return face_names