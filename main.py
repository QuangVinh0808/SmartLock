import cv2

from core.preprocess import ImagePreprocessor
from core.detector import HOGFaceDetector

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
    

