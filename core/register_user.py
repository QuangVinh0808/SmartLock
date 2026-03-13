import cv2

from core.detector import HOGFaceDetector
from core.recognizer import MobileFaceNetRecognizer
from core.database import FaceDatabase


detector = HOGFaceDetector()

recognizer = MobileFaceNetRecognizer(
    model_path="models/MobileFaceNet.onnx"
)

database = FaceDatabase("data/face_db.pkl")


name = input("Enter user name: ")

cap = cv2.VideoCapture(0)

embeddings = []

count = 0
max_samples = 30


while count < max_samples:

    ret, frame = cap.read()

    if not ret:
        break

    boxes = detector.detect(frame)

    faces = detector.crop_faces(frame, boxes)

    if len(faces) > 0:

        face = faces[0]

        emb = recognizer.get_embedding(face)

        embeddings.append(emb)

        count += 1

        print("Captured:", count)

    cv2.imshow("Register User", frame)

    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()


# lưu vào database
database.add_user(name, embeddings)

print("User registered:", name)