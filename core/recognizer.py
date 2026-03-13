import cv2
import numpy as np
import pickle
import os


class MobileFaceNetRecognizer:

    def __init__(self,
                 model_path="mobilefacenet.onnx",
                 db_path="face_db.pkl",
                 threshold=0.9):

        self.net = cv2.dnn.readNetFromONNX(model_path)

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

        face = cv2.resize(face, (112, 112))

        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        face = face.astype(np.float32)

        face = (face - 127.5) / 128.0

        face = np.transpose(face, (2, 0, 1))

        face = np.expand_dims(face, axis=0)

        return face

    # =========================================
    # EXTRACT EMBEDDING
    # =========================================
    def get_embedding(self, face):

        blob = self.preprocess(face)

        self.net.setInput(blob)

        emb = self.net.forward().flatten()

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