import csv

import numpy as np
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image


class FeatureExtractor:
  def __init__(self, base_route, model_name, output_route, dim=139, bath_size=128):
    
    self.bath_size = bath_size
    self.__width = self.__height = dim
    self.image_shape = (self.width, self.height, 3)
    
    self.model_name = model_name
    self.train_route = base_route
    self.output_route = output_route
    
    self.labels = []
    self.features = []
    
    self.model = None
    self.directory_iterator = None
  
  def get_directory_iterator(self, route):
    image_generator = image.ImageDataGenerator(rescale=1.0 / 255)
    
    self.directory_iterator = image_generator.flow_from_directory(
      directory=route,
      target_size=(self.width, self.height),
      batch_size=self.bath_size,
      class_mode="categorical"
    )
  
  def set_extractor_model(self):
    if self.model_name == "inception_v3":
      self.model = InceptionV3(include_top=False, weights="imagenet", input_shape=self.image_shape)
    elif self.model_name == "inception_resnet_v2":
      self.model = InceptionResNetV2(include_top=False, weights="imagenet", input_shape=self.image_shape)
    elif self.model_name == "vgg19":
      self.model = VGG19(include_top=False, weights="imagenet", input_shape=self.image_shape)
    else:
      raise Exception("Invalid pre-trained Keras Application")
  
  def load_labels(self):
    for _ in range(self.directory_iterator.samples):
      _, y = next(self.directory_iterator)
      self.labels.append(y.argmax())
  
  def load_features(self):
    features = self.model.predict_generator(self.directory_iterator)
    
    for feature in features:
      self.features.append(feature.flatten())
  
  def extract(self):
    self.load_labels()
    self.load_features()
  
  def save_csv(self):
    self.features = np.array(self.features)
    self.labels = np.array(self.labels)
    
    number_of_instances = self.features.shape[0]
    number_of_features = self.features.shape[-1]
    
    with open(self.output_route, 'w') as f:
      writer = csv.writer(f)
      
      headers = [i for i in range(number_of_features)]
      headers.append(-1)
      writer.writerow(headers)
      
      for i in range(number_of_instances):
        features = self.features[i]
        row = np.append(features, [self.labels[i]])
        writer.writerow(row)
  
  def extract_and_store(self):
    
    self.get_directory_iterator(self.train_route)
    self.set_extractor_model()
    
    print("Extracting...")
    self.extract()
    
    print("Saving...")
    self.save_csv()
  
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
