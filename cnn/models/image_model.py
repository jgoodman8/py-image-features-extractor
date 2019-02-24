import os
import math

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image


class ImageModel:
  def __init__(self, base_route, train_folder="train", validation_folder="validation"):
    self.width = self.height = 64
    self.train_route = os.path.join(base_route, train_folder)
    self.validation_route = os.path.join(base_route, validation_folder)
    
    self.epochs = 1000
    self.__batch_size = 256
    
    self.model_route = "model.h5"
    self.early = ImageModel.get_early_stop()
    self.checkpoint = self.get_model_checkpoint()
    
    self.model = None
    self.__train_size = 0
    self.__validation_size = 0
    self.train_steps = 0
    self.validation_steps = 0
  
  def build_model(self, number_of_classes):
    # create the base pre-trained model
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(self.width, self.height, 3))
    
    for layer in base_model.layers:
      layer.trainable = False
    
    # add a global spatial average pooling layer
    x = base_model.output
    
    x = Flatten()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(number_of_classes, activation='softmax')(x)
    
    # this is the model we will train
    self.model = Model(inputs=base_model.input, outputs=predictions)
    
    self.model.compile(
      optimizer=Adam(),
      loss='categorical_crossentropy',
      metrics=["accuracy"]
    )
    
  def fine_tune(self):
    for layer in self.model.layers:
      layer.trainable = True
      
    self.model.compile(
      optimizer=Adam(),
      loss='categorical_crossentropy',
      metrics=["accuracy"]
    )
  
  def train(self):
    train_directory_iterator = ImageModel.get_directory_iterator(self.train_route)
    self.train_size = train_directory_iterator.samples
    
    validation_directory_iterator = ImageModel.get_directory_iterator(self.validation_route)
    self.validation_size = validation_directory_iterator.samples
    
    self.build_model(train_directory_iterator.num_classes)
    
    self.model.fit_generator(
      train_directory_iterator,
      steps_per_epoch=self.train_steps,
      epochs=self.epochs,
      validation_data=validation_directory_iterator,
      validation_steps=self.validation_steps,
      callbacks=[self.checkpoint, self.early]
    )
    
    self.fine_tune()
    
    self.model.fit_generator(
      train_directory_iterator,
      steps_per_epoch=self.train_steps,
      epochs=self.epochs,
      validation_data=validation_directory_iterator,
      validation_steps=self.validation_steps,
      callbacks=[self.checkpoint, self.early]
    )
    
    self.model.save("vgg19_model.h5")
    
    return self.model.evaluate_generator(train_directory_iterator)
  
  def get_model_checkpoint(self):
    return ModelCheckpoint(
      self.model_route,
      monitor='val_acc',
      verbose=1,
      save_best_only=True,
      save_weights_only=False,
      mode='auto',
      period=1
    )
  
  @staticmethod
  def get_early_stop():
    return EarlyStopping(
      monitor='val_acc',
      min_delta=0,
      patience=10,
      verbose=1,
      mode='auto'
    )

  @staticmethod
  def get_directory_iterator(route):
    image_generator = image.ImageDataGenerator(rescale=1.0 / 255)
  
    return image_generator.flow_from_directory(
      directory=route,
      target_size=(64, 64),
      batch_size=64,
      class_mode="categorical"
    )
  
  @property
  def train_size(self):
    return self.__train_size
  
  @train_size.setter
  def train_size(self, train_size):
    self.__train_size = train_size
    self.train_steps = math.ceil(self.train_size / self.batch_size)
  
  @property
  def validation_size(self):
    return self.__validation_size
  
  @validation_size.setter
  def validation_size(self, validation_size):
    self.__validation_size = validation_size
    self.validation_steps = math.ceil(self.validation_size / self.batch_size)
  
  @property
  def batch_size(self):
    return self.__batch_size
  
  @batch_size.setter
  def batch_size(self, batch_size):
    self.__batch_size = batch_size
    self.train_steps = math.ceil(self.train_size / self.batch_size)
    self.validation_steps = math.ceil(self.validation_size / self.batch_size)
