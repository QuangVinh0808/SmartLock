import numpy as np
import cv2
import pickle
import os
import tensorflow as tf


class FaceNetRecognizer:

    def __init__(self,
                 model_path="models/facenet.tflite",
                 db_path="data/face_db.pkl",
                 threshold=0.8):

        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Load database
        self.db_path = db_path

        if os.path.exists(db_path):
            with open(db_path, "rb") as f:
                self.database = pickle.load(f)
        else:
            self.database = {}

        self.threshold = threshold

    # =========================================
    # PREPROCESS
    # =========================================
    def preprocess(self, face):

        face = cv2.resize(face, (160, 160))

        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        face = face.astype(np.float32)

        # chuẩn hóa giống FaceNet
        mean, std = face.mean(), face.std()
        face = (face - mean) / std

        face = np.expand_dims(face, axis=0)

        return face

    # =========================================
    # GET EMBEDDING
    # =========================================
    def get_embedding(self, face):

        input_data = self.preprocess(face)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        self.interpreter.invoke()

        embedding = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        # normalize vector
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    # =========================================
    # RECOGNIZE
    # =========================================
    def recognize(self, face):

        emb = self.get_embedding(face)

        best_name = "Unknown"
        best_score = -1  # cosine similarity

        for name, vectors in self.database.items():

            for db_emb in vectors:

                db_emb = np.array(db_emb)

                # cosine similarity
                score = np.dot(emb, db_emb)

                if score > best_score:
                    best_score = score
                    best_name = name

        if best_score < self.threshold:
            best_name = "Unknown"

        return best_name, emb, best_score

    # =========================================
    # ADAPTIVE UPDATE
    # =========================================
    def adaptive_update(self, name, emb):

        if name == "Unknown":
            return

        if name not in self.database:
            self.database[name] = []

        self.database[name].append(emb.tolist())

        # giữ max 50 embeddings
        if len(self.database[name]) > 50:
            self.database[name] = self.database[name][-50:]

        self.save()

    # =========================================
    # SAVE DB
    # =========================================
    def save(self):

        with open(self.db_path, "wb") as f:
            pickle.dump(self.database, f)