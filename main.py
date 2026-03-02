from core.preprocess import ImagePreprocessor



if __name__ == "__main__":
    preprocessor = ImagePreprocessor(gamma=1.2, clip_limit=2.0, tile_grid_size=(8, 8))
    

