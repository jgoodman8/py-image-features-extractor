import math

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image


class ImageModel:
  def __init__(self, base_route):
    self.width = self.height = 64
    self.train_route = base_route + "/train"
    self.validation_route = base_route + "/validation"
    
    self.epochs = 1000
    self.batch_size = 256
    self.train_size = 100000
    self.validation_size = 10000
    self.steps_per_epoch = math.ceil(self.train_size / self.batch_size)
    self.validation_steps = math.ceil(self.validation_size / self.batch_size)
    
    self.early = EarlyStopping(
      monitor='val_acc',
      min_delta=0,
      patience=10,
      verbose=1,
      mode='auto'
    )
    
    self.checkpoint = ModelCheckpoint(
      "vgg16.h5",
      monitor='val_acc',
      verbose=1,
      save_best_only=True,
      save_weights_only=False,
      mode='auto',
      period=1
    )
    
    self.model = None
  
  def get_directory_iterator(self, route):
    image_generator = image.ImageDataGenerator(rescale=1.0 / 255)
    
    return image_generator.flow_from_directory(
      directory=route,
      target_size=(self.width, self.height),
      batch_size=self.batch_size,
      class_mode="categorical"
    )
  
  def set_model(self):
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
    predictions = Dense(200, activation='softmax')(x)
    
    # this is the model we will train
    self.model = Model(inputs=base_model.input, outputs=predictions)
  
  def train(self):
    self.set_model()
    self.model.summary()
    self.model.compile(
      optimizer=Adam(),
      loss='categorical_crossentropy',
      metrics=["accuracy"]
    )
    
    train_directory_iterator = self.get_directory_iterator(self.train_route)
    validation_directory_iterator = self.get_directory_iterator(self.validation_route)
    
    self.model.fit_generator(
      train_directory_iterator,
      steps_per_epoch=self.steps_per_epoch,
      epochs=self.epochs,
      validation_data=validation_directory_iterator,
      validation_steps=self.validation_steps,
      callbacks=[self.checkpoint, self.early]
    )
    
    return self.model.evaluate_generator(train_directory_iterator)
