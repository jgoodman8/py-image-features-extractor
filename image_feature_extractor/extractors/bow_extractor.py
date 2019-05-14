import os
from typing import List, Dict

import cv2
import numpy as np

from image_feature_extractor.extractors.extractor import Extractor
from image_feature_extractor.models import ClusteringModel


class BoWExtractor(Extractor):
    
    def __init__(self, base_route: str, method: str, k: int = 0, min_k: int = 0, max_k: int = 0, step: int = 0,
                 threshold: float = 0.85, cluster_mode: str = 'manual', size: int = 224, batch_size: int = 128,
                 color_code: int = cv2.COLOR_BGR2RGB):
        
        super().__init__(base_route=base_route, size=size, batch_size=batch_size)
        
        self.k: int = k
        self.image_keypoints: Dict = {}
        
        self.detector = None
        self.bow_kmeans = None
        self.__bow_extractor = None
        self.__empty_histogram: List[float] = [0.0] * self.k
        self.__color_code: int = color_code
        
        self._set_detector(method)
        self._set_clustering_trainer(mode=cluster_mode, min_k=min_k, max_k=max_k, step=step, threshold=threshold)
    
    def setup(self):
        for filename, category in zip(self.directory_iterator.filenames, self.directory_iterator.classes):
            image_route = os.path.join(self.base_route, filename)
            self._extract_and_add_image(image_route)
        
        self._update_counter()
    
    def fit(self):
        vocabulary = self.bow_kmeans.cluster()
        self.__bow_extractor = cv2.BOWImgDescriptorExtractor(self.detector, cv2.BFMatcher(cv2.NORM_L2))
        self.__bow_extractor.setVocabulary(vocabulary)
    
    def extract(self, image_route: str):
        image = self._read_image(image_route)
        keypoints = self.image_keypoints[image_route]
        if keypoints is None or len(keypoints) == 0:
            return self.__empty_histogram
        
        return self.__bow_extractor.compute(image, keypoints)[0]
    
    def _set_detector(self, method: str):
        # TODO: Add more detection methods
        if method.lower() == 'kaze':
            self.detector = cv2.KAZE_create()
        elif method.lower() == 'orb':
            self.detector = cv2.ORB_create()
        elif method.lower() == 'akaze':
            self.detector = cv2.AKAZE_create()
    
    def _set_clustering_trainer(self, mode: str, min_k: int = 0, max_k: int = 0, step: int = 0,
                                threshold: float = 0.85):
        if mode == 'manual':
            self.bow_kmeans = cv2.BOWKMeansTrainer(self.k)
        elif mode == 'auto':
            self.bow_kmeans = ClusteringModel(min_k, max_k, step, threshold)
    
    def _extract_and_add_image(self, image_route: str):
        keypoints, descriptors = self._process_image(image_route)
        self._add_image(image_route, keypoints, descriptors)
    
    def _process_image(self, image_route: str) -> (List[cv2.KeyPoint], np.ndarray):
        return self.detector.detectAndCompute(self._read_image(image_route), None)
    
    def _read_image(self, image_route: str):
        return cv2.cvtColor(cv2.imread(image_route), self.__color_code)
    
    def _add_image(self, image_route: str, keypoints: List[cv2.KeyPoint], descriptors: np.ndarray):
        if descriptors is not None and len(descriptors) > 0:
            self.bow_kmeans.add(descriptors)
        self.image_keypoints[image_route] = keypoints
    
    def _find_features_size(self) -> int:
        return self.k
