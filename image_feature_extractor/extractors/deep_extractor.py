import os

import numpy as np
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as resnet_preprocessor
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocessor
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocessor
from tensorflow.keras.preprocessing import image as image_preprocessor

from image_feature_extractor.extractors.extractor import Extractor


class DeepExtractor(Extractor):
    
    def __init__(self, base_route, model_name: str, size=224, batch_size=128):
        
        super().__init__(base_route=base_route, size=size, batch_size=batch_size)
        
        self.model = None
        self.file_writer = None
        self.model_preprocess = None
        self.model_name = model_name
        
        self._set_extractor_model(self.model_name)
    
    def _set_extractor_model(self, model_name: str) -> None:
        if model_name == "inception_v3":
            self.model = InceptionV3(include_top=False, weights="imagenet", input_shape=self.image_shape)
            self.model_preprocess = inception_preprocessor
        elif model_name == "inception_resnet_v2":
            self.model = InceptionResNetV2(include_top=False, weights="imagenet", input_shape=self.image_shape)
            self.model_preprocess = resnet_preprocessor
        elif model_name == "vgg19":
            self.model = VGG19(include_top=False, weights="imagenet", input_shape=self.image_shape)
            self.model_preprocess = vgg19_preprocessor
        else:
            raise Exception("Invalid pre-trained Keras Application")
    
    def extract(self, image_route: str) -> np.ndarray:
        image = image_preprocessor.load_img(image_route, target_size=(self.width, self.height))
        image = np.expand_dims(image_preprocessor.img_to_array(image), axis=0)
        preprocessed_img = self.model_preprocess(image)
        
        return self.model.predict(preprocessed_img).flatten()
    
    def _find_features_size(self) -> int:
        example_image_route = os.path.join(self.base_route, self.directory_iterator.filenames[0])
        return len(self.extract(image_route=example_image_route))
