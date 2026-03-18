import cv2

from core.preprocess import ImagePreprocessor
from core.detector import HOGFaceDetector
from core.recognizer import FaceNetTFLiteRecognizer

#Configure instance of PiCamera2
"""
cam = Picamera2()
## Set the resolution of the camera preview
cam.preview_configuration.main.size = (640, 360)
cam.preview_configuration.main.format = "RGB888"
cam.preview_configuration.controls.FrameRate=30
cam.preview_configuration.align()
cam.configure("preview")
cam.start()
"""
video_capture = cv2.VideoCapture(0) # Sử dụng webcam thay vì PiCamera2
if __name__ == "__main__":
    preprocessor = ImagePreprocessor(gamma=1.5, clip_limit=2.0, tile_grid_size=(8, 8))
    detector = HOGFaceDetector()
    recognizer = FaceNetTFLiteRecognizer(model_path="models/facenet.tflite", db_path="face_db.pkl", threshold=0.9)

    # Ví dụ nhận diện khuôn mặt
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frame_pp = preprocessor.preprocess(frame)
        boxes = detector.detect(frame_pp)
        for box in boxes:
            x, y, w, h = box
            face = frame_pp[y:y+h, x:x+w]
            name, emb = recognizer.recognize(face)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("SmartLock", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
    

