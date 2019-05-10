import os
from typing import List, Tuple

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

from image_feature_extractor.extractors.extractor import Extractor


class LBPExtractor(Extractor):
    
    def __init__(self, route: str, output_file: str, points: int, radius: int, size: int, grid_x: int, grid_y: int,
                 verbose: int = 1, verbosity_frequency: int = 10, image_extension: str = 'JPEG', header: int = 1,
                 method: str = 'uniform', color_code: int = cv2.COLOR_BGR2GRAY):
        
        super().__init__(route=route, output_file=output_file, verbose=verbose, verbosity_frequency=verbosity_frequency,
                         image_extension=image_extension, header=header)
        
        self.points: int = points
        self.radius: int = radius
        self.method: str = method
        self.color_code: int = color_code
        
        self.bins: np.ndarray = np.arange(0, self.points + 3)
        self.range: Tuple[int, int] = (0, self.points + 2)
        
        self.width: int = size
        self.height: int = size
        self.height_step: int = self.height // grid_y
        self.width_step: int = self.width // grid_x
    
    def extract(self, image_route: str) -> np.ndarray:
        """
        Generates an histogram for the given image route.
        
        :param image_route:
        :return:
        """
        
        image = self._read_image(image_route)
        lbp = self._extract_lbp(image)
        
        histogram_grid = self._convert_lbp_to_histogram(lbp)
        
        return np.array(histogram_grid).flatten()
    
    def _read_image(self, image_route: str) -> np.ndarray:
        image = cv2.imread(image_route)
        gray_image = cv2.cvtColor(image, self.color_code)
        
        return gray_image
    
    def _extract_lbp(self, image: np.ndarray) -> np.ndarray:
        return local_binary_pattern(image, self.points, self.radius, method=self.method)
    
    def _convert_lbp_to_histogram(self, image: np.ndarray) -> List[np.ndarray]:
        grid = []
        
        for y in range(0, self.height, self.height_step):
            for x in range(0, self.width, self.width_step):
                y_limit = y + self.height_step
                x_limit = x + self.width_step
                
                batch = image[y:y_limit, x:x_limit]
                
                grid.append(self._convert_image_chunk_to_histogram(batch))
        
        return grid
    
    def _convert_image_chunk_to_histogram(self, lbp_chunk: np.ndarray) -> np.ndarray:
        hist, _ = np.histogram(lbp_chunk, bins=self.bins, range=self.range)
        
        return hist
    
    def _find_features_size(self) -> int:
        folder = os.listdir(self.base_route)[0]
        images_folder = self._find_images_folder(folder)
        features = self.extract(images_folder)
        
        return len(features)
