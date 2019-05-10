import csv
import os
import sys

import math
import numpy as np


class Extractor:
    
    def __init__(self, route: str, output_file: str, verbose: int = 1, verbosity_frequency: int = 10,
                 image_extension: str = 'JPEG', header: int = 1):
        
        self.base_route: str = route
        self.output_file: str = output_file
        self.num_classes: int = len(os.listdir(route))
        self.file_writer = None
        
        self.counter: int = 0
        self.verbose: int = verbose
        self.header: int = header
        self.verbosity_frequency: int = verbosity_frequency
        self.image_extension: str = image_extension
    
    def extract_and_save(self):
        
        with open(self.output_file, 'w') as f:
            self.file_writer = csv.writer(f)
            self._print_header()
            
            for folder_name in os.listdir(self.base_route):
                folder_route = self._find_images_folder(folder_name)
                
                for image_name in os.listdir(folder_route):
                    image_route = os.path.join(folder_route, image_name)
                    
                    features = self.extract(image_route=image_route)
                    self._print_row(features, folder_name)
                
                self._update_counter()
    
    def _print_header(self):
        if self.header:
            headers = list(range(self._find_features_size()))
            headers.append(-1)
            self.file_writer.writerow()
    
    def _print_row(self, features: np.ndarray, label) -> None:
        self.file_writer.writerow(np.append(features, [label]))
    
    def _find_images_folder(self, folder: str) -> str:
        folder_content = os.listdir(folder)
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
