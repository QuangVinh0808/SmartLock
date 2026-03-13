import pickle
import os


class FaceDatabase:

    def __init__(self, db_path="data/face_db.pkl"):

        self.db_path = db_path

        # nếu file tồn tại thì load
        if os.path.exists(self.db_path):

            with open(self.db_path, "rb") as f:
                self.data = pickle.load(f)

        else:
            self.data = {}

    # =====================================
    # LƯU DATABASE
    # =====================================
    def save(self):

        with open(self.db_path, "wb") as f:
            pickle.dump(self.data, f)

    # =====================================
    # THÊM USER MỚI
    # =====================================
    def add_user(self, name, embeddings):

        if name not in self.data:
            self.data[name] = []

        self.data[name].extend(embeddings)

        self.save()

    # =====================================
    # LẤY TẤT CẢ DATA
    # =====================================
    def get_all(self):

        return self.data

    # =====================================
    # LẤY EMBEDDING CỦA 1 USER
    # =====================================
    def get_user(self, name):

        return self.data.get(name, [])

    # =====================================
    # XOÁ USER
    # =====================================
    def remove_user(self, name):

        if name in self.data:
            del self.data[name]
            self.save()

    # =====================================
    # DANH SÁCH USER
    # =====================================
    def list_users(self):

        return list(self.data.keys())