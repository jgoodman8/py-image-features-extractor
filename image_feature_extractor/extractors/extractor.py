import csv
import os
import sys
from abc import ABC

import math
import numpy as np


class Extractor(ABC):
    
    def __init__(self, base_route: str, image_extension: str = 'JPEG'):
        
        self.base_route: str = base_route
        self.output_file: str = ''
        self.num_classes: int = len(os.listdir(base_route))
        self.file_writer = None
        
        self.counter: int = 0
        self.verbose: int = 1
        self.header: int = 1
        self.verbosity_frequency: int = 10
        self.image_extension: str = image_extension
        super().__init__()
    
    def extract_and_save(self, output_file: str, verbose: int = 1, verbosity_frequency: int = 10, header: int = 1):
        self.output_file = output_file
        self.header: int = header
        self.verbose: int = verbose
        self.verbosity_frequency: int = verbosity_frequency
        
        with open(self.output_file, 'w') as f:
            self.file_writer = csv.writer(f)
            self._print_header()
            
            for folder_name in os.listdir(self.base_route):
                folder_route = self._find_images_folder(folder_name)
                for image_name in os.listdir(folder_route):
                    features = self._extract_image_features(folder_route, image_name)
                    self._print_row(features, folder_name)
                
                self._update_counter()
    
    def _extract_image_features(self, folder_route: str, image_name: str) -> np.ndarray:
        image_route = os.path.join(folder_route, image_name)
        return self.extract(image_route=image_route)
    
    def _print_header(self):
        if self.header:
            header = list(range(self._find_features_size()))
            header.append(-1)
            self.file_writer.writerow(header)
    
    def _print_row(self, features: np.ndarray, label) -> None:
        self.file_writer.writerow(np.append(features, [label]))
    
    def _find_images_folder(self, folder: str) -> str:
        folder_content = os.listdir(os.path.join(self.base_route, folder))
        if folder_content is not None and len(folder_content) and self.image_extension in folder_content[0]:
            return os.path.join(self.base_route, folder)
        
        return os.path.join(self.base_route, folder, 'images')
    
    def _update_counter(self) -> None:
        self.counter += 1
        
        if self.verbose and (self.counter % self.verbosity_frequency == 0):
            progress_percent: int = math.floor((self.counter / self.num_classes) * 100)
            sys.stdout.write("{}%".format(progress_percent))
    
    def _find_features_size(self) -> int:
        pass
    
    def extract(self, image_route: str) -> np.ndarray:
        pass
