import cv2
import cv2
import numpy as np
import pickle
import os
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

class MobileFaceNetRecognizer:
class FaceNetTFLiteRecognizer:
    def __init__(self,
    def __init__(self,
                 model_path="../models/facenet.tflite",
                 db_path="face_db.pkl",
                 threshold=0.9):
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.db_path = db_path
        self.threshold = threshold

        if os.path.exists(db_path):
            with open(db_path, "rb") as f:
                self.database = pickle.load(f)
        else:
            self.database = {}

    # =========================================
    # FACE PREPROCESSING
    # =========================================
    def preprocess(self, face):
    def preprocess(self, face):
        face = cv2.resize(face, (160, 160))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype(np.float32)
        face = (face - 127.5) / 128.0
        face = np.expand_dims(face, axis=0)
        return face

    # =========================================
    # EXTRACT EMBEDDING
    # =========================================
    def get_embedding(self, face):
        input_data = self.preprocess(face)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        emb = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        emb = emb / np.linalg.norm(emb)
        return emb

    # =========================================
    # RECOGNITION
    # =========================================
    def recognize(self, face):
        emb = self.get_embedding(face)
        best_name = "Unknown"
        best_dist = 999
        for name, vectors in self.database.items():
            for v in vectors:
                dist = np.linalg.norm(emb - v)
                if dist < best_dist:
                    best_dist = dist
                    best_name = name
        if best_dist > self.threshold:
            best_name = "Unknown"
        return best_name, emb
        return best_name, emb

    # =========================================
    # REGISTER NEW USER
    # =========================================
    def register(self, name, faces):

        embeddings = []

        for f in faces:

            emb = self.get_embedding(f)

            embeddings.append(emb)

        self.database[name] = embeddings

        self.save()

    # =========================================
    # ADAPTIVE LEARNING
    # =========================================
    def adaptive_update(self, name, emb):

        if name == "Unknown":
            return

        self.database[name].append(emb)

        if len(self.database[name]) > 50:
            self.database[name] = self.database[name][-50:]

        self.save()

    def save(self):

        with open(self.db_path, "wb") as f:
            pickle.dump(self.database, f)