import csv
import os
import sys
from abc import ABC

import math
import numpy as np
from tensorflow.python.keras.preprocessing import image as image_preprocessor


class Extractor(ABC):
    
    def __init__(self, base_route: str, size=224, batch_size=128):
        
        self.base_route: str = base_route
        self.output_file: str = ''
        self.num_classes: int = len(os.listdir(base_route))
        self.file_writer = None
        
        self.bath_size = batch_size
        self.__width = self.__height = size
        self.image_shape = (self.width, self.height, 3)
        self.directory_iterator = None
        
        self.counter: int = 0
        self.verbose: int = 1
        self.header: int = 1
        self.verbosity_frequency: int = 10
        
        self._set_directory_iterator(self.base_route)
        super().__init__()
    
    def extract_and_save(self, output_file: str, verbose: int = 1, verbosity_frequency: int = 10, header: int = 1):
        self.output_file = output_file
        self.header: int = header
        self.verbose: int = verbose
        self.verbosity_frequency: int = verbosity_frequency
        
        with open(self.output_file, 'w') as f:
            self.file_writer = csv.writer(f)
            self._print_header()
            
            for filename, category in zip(self.directory_iterator.filenames, self.directory_iterator.classes):
                image_route = os.path.join(self.base_route, filename)
                features = self.extract(image_route=image_route)
                self._print_row(features=features, label=category)
            
            self._update_counter()
    
    def extract(self, image_route: str) -> np.ndarray:
        pass
    
    def _set_directory_iterator(self, route: str) -> None:
        image_generator = image_preprocessor.ImageDataGenerator(rescale=1.0 / 255)
        
        self.directory_iterator = image_generator.flow_from_directory(
            directory=route,
            target_size=(self.width, self.height),
            batch_size=self.bath_size,
            class_mode="categorical"
        )
    
    def _print_row(self, features: np.ndarray, label) -> None:
        self.file_writer.writerow(np.append(features, [label]))
    
    def _print_header(self):
        if self.header:
            column_names = list(range(self._find_features_size())) + [-1]
            self.file_writer.writerow(column_names)
    
    def _find_features_size(self) -> int:
        pass
    
    def _update_counter(self) -> None:
        self.counter += 1
        
        if self.verbose and (self.counter % self.verbosity_frequency == 0):
            progress_percent: int = math.floor((self.counter / self.num_classes) * 100)
            sys.stdout.write("{}%".format(progress_percent))
    
    def _reset_counter(self) -> None:
        self.counter = 0
    
    @property
    def width(self):
        return self.__width
    
    @width.setter
    def width(self, width):
        self.__width = width
        self.image_shape = (width, self.height, 3)
    
    @property
    def height(self):
        return self.__height
    
    @height.setter
    def height(self, height):
        self.__height = height
        self.image_shape = (self.width, height, 3)
