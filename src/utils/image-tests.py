# External
import os
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.measure import shannon_entropy

class ImageTests:

    def __init__(self, image_path: str) -> None:
        self.image_path = image_path
        self.MAX_IMAGES = 50

if "__main__" == __name__:
    it = ImageTests("data/RQ3_TWPA_V2_W2/251023_Junctions/bright")
    
    

