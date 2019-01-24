import math

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import SGD


class ImageModel:
  def __init__(self, base_route):
    self.width = self.height = 64
    self.train_route = base_route + "/train"
    
    self.early = EarlyStopping(
      monitor='val_acc',
      min_delta=0,
      patience=10,
      verbose=1,
      mode='auto'
    )
    
    self.checkpoint = ModelCheckpoint(
      "inception.h5",
      monitor='val_acc',
      verbose=1,
      save_best_only=True,
      save_weights_only=False,
      mode='auto',
      period=1
    )
    
    self.model = None
    self.base_layers_size = 0
  
  def get_directory_iterator(self, route):
    image_generator = image.ImageDataGenerator(rescale=1.0 / 255)
    
    return image_generator.flow_from_directory(
      directory=route,
      target_size=(self.width, self.height),
      batch_size=16,
      class_mode=None,
      shuffle=False
    )
  
  def set_model(self):
    # create the base pre-trained model
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(self.width, self.height, 3))
    
    self.base_layers_size = len(base_model.layers)
    
    # add a global spatial average pooling layer
    x = base_model.output
    
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(200, activation='softmax')(x)
    
    # this is the model we will train
    self.model = Model(inputs=base_model.input, outputs=predictions)
  
  def train(self):
    epochs = 10
    batch_size = 256
    train_size = 100000
    validation_size = 10000
    steps_per_epoch = math.ceil(train_size / batch_size)
    validation_steps = math.ceil(validation_size / batch_size)
    
    self.set_model()
    self.model.summary()
    self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    
    directory_iterator = self.get_directory_iterator(self.train_route)
    
    self.model.fit_generator(
      directory_iterator,
      steps_per_epoch=steps_per_epoch,
      epochs=10,
      # validation_data=validation_generator,
      validation_steps=validation_steps,
      callbacks=[self.checkpoint, self.early]
    )
    
    for layer in self.model.layers[:self.base_layers_size]:
      layer.trainable = False
    for layer in self.model.layers[self.base_layers_size:]:
      layer.trainable = True
    
    self.model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
    
    self.model.fit_generator(
      directory_iterator,
      steps_per_epoch=steps_per_epoch,
      epochs=epochs,
      # validation_data=validation_generator,
      validation_steps=validation_steps,
      callbacks=[self.checkpoint, self.early]
    )
    
    return self.model.evaluate_generator(directory_iterator)
