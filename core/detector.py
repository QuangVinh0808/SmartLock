import cv2
import dlib


class HOGFaceDetector:


    def __init__(self,
                 resize_width=320,
                 upsample_times=0,
                 detect_every_n_frames=2):
        """
        Parameters
        ----------
        resize_width : int
            Resize width before detection (speed optimization)

        upsample_times : int
            Upsample image for detecting smaller faces

        detect_every_n_frames : int
            Run detection every N frames (cache between frames)
        """

        self.detector = dlib.get_frontal_face_detector()

        self.resize_width = resize_width
        self.upsample_times = upsample_times
        self.detect_every_n_frames = detect_every_n_frames

        self.frame_count = 0
        self.last_boxes = []

    # =========================================================
    # MAIN DETECTION FUNCTION
    # =========================================================
    def detect(self, frame):
        self.frame_count += 1

        # Skip frame nếu chưa đến lượt detect
        if self.frame_count % self.detect_every_n_frames != 0:
            return self.last_boxes

        h, w = frame.shape[:2]

        scale = self.resize_width / float(w)
        new_h = int(h * scale)

        small = cv2.resize(frame, (self.resize_width, new_h))

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        faces = self.detector(gray, self.upsample_times)

        boxes = []

        for face in faces:

            x1 = int(face.left() / scale)
            y1 = int(face.top() / scale)
            x2 = int(face.right() / scale)
            y2 = int(face.bottom() / scale)

            x1 = max(0, x1)
            y1 = max(0, y1)

            boxes.append((x1, y1, x2, y2))

        self.last_boxes = boxes
        return boxes

    # =========================================================
    # CROP FACE ROI
    # =========================================================
    def crop_faces(self, frame, boxes):
        """
        Extract face regions from bounding boxes
        """

        faces = []
        h, w = frame.shape[:2]

        for (x1, y1, x2, y2) in boxes:

            x2 = min(w, x2)
            y2 = min(h, y2)

            roi = frame[y1:y2, x1:x2]

            if roi.size != 0:
                faces.append(roi)

        return faces

    # =========================================================
    # DRAW BOUNDING BOXES (DEBUG)
    # =========================================================
    def draw_boxes(self, frame, boxes, color=(0, 255, 0)):
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        return frame

    # =========================================================
    # RESET DETECTOR STATE
    # =========================================================
    def reset(self):
        self.frame_count = 0
        self.last_boxes = []