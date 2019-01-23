from tensorflow.keras.layers import Input
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image


class ImageModel:
  def __init__(self, base_route):
    self.width = self.height = 64
    self.train_route = base_route + "/train"
  
  def get_directory_iterator(self, route):
    image_generator = image.ImageDataGenerator(rescale=1.0 / 255)
    
    return image_generator.flow_from_directory(
      directory=route,
      target_size=(self.width, self.height),
      batch_size=16,
      class_mode=None,
      shuffle=False
    )
  
  def train(self):
    input_tensor = Input(shape=(self.width, self.height, 3))
    model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=True, classes=200)  # TODO: wrong instance
    
    model.compile()
    
    directory_iterator = self.get_directory_iterator(self.train_route)
    
    return model.evaluate_generator(directory_iterator)
