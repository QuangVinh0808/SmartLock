
import cv2
import numpy as np

class ImagePreprocessor:
    def __init__(self, gamma=1.0, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.gamma = gamma
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )
        #Tạo LUT cho gamma correction
        if gamma != 1.0:
            inv_gamma = 1.0 / gamma
            self.gamma_table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        else:
            self.gamma_table = None
    
    def apply_gamma_correction(self, image):
        if self.gamma_table is not None:
            return cv2.LUT(image, self.gamma_table)
        return image
    
    def apply_clahe_lab(self, image):
        if len(image.shape) == 2 and image.shape[2] == 1: # Kiểm tra nếu ảnh có 3 kênh màu
            lab = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else: 
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l) #Chỉ áp dụng CLAHE lên kênh L
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def preprocess(self, frame):
        frame = self.apply_gamma_correction(frame)
        frame = self.apply_clahe_lab(frame)
        return frame