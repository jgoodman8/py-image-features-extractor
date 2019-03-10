import math
import os

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing import image


class ImageModel:
    def __init__(self, base_route, train_folder="train", validation_folder="val", epochs=10, fine_tune: bool = False):
        self.__model = None
        self.__base_model = None
        self.__width = self.__height = 64
        self.__train_route = os.path.join(base_route, train_folder)
        self.__validation_route = os.path.join(base_route, validation_folder)
        
        self.__fine_tuning = fine_tune
        
        self.__epochs = epochs
        self.__batch_size = 256
        
        self.__model_route = "model.h5"
        self.__early_stop = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
        self.__checkpoint = self._get_model_checkpoint()
        
        self.train_size = 0
        self.validation_size = 0
        self.train_steps = 0
        self.validation_steps = 0
    
    def train(self):
        train_directory_iterator = self._get_directory_iterator(self.__train_route)
        validation_directory_iterator = self._get_directory_iterator(self.__validation_route)
        
        self.train_size = train_directory_iterator.samples
        self.validation_size = validation_directory_iterator.samples
        
        self._build_model(train_directory_iterator.num_classes)
        
        if self.__fine_tuning:
            self._set_fine_tune()
        else:
            self._set_transfer_learning()
        
        self.__model.fit_generator(
            train_directory_iterator,
            steps_per_epoch=self.train_steps,
            epochs=self.__epochs,
            validation_data=validation_directory_iterator,
            validation_steps=self.validation_steps,
            callbacks=[self.__checkpoint, self.__early_stop]
        )
        
        self.__model.save(self.__model_route)
        
        metrics = self.__model.evaluate_generator(train_directory_iterator)
        
        return metrics
    
    def _build_model(self, num_classes: int):
        self.__base_model = VGG19(weights='imagenet', include_top=False, input_shape=(self.__width, self.__height, 3))
        
        x = self.__base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        
        self.__model = Model(inputs=self.__base_model.input, outputs=predictions)
    
    def _set_transfer_learning(self):
        for layer in self.__base_model.layers:
            layer.trainable = False
        
        self.__model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    
    def _set_fine_tune(self):
        layers_to_freeze = int(len(self.__model.layers) * 0.9)
        
        for layer in self.__model.layers[:layers_to_freeze]:
            layer.trainable = False
        for layer in self.__model.layers[layers_to_freeze:]:
            layer.trainable = True
        
        self.__model.compile(
            optimizer=SGD(lr=0.0001, momentum=0.9),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def _get_model_checkpoint(self):
        return ModelCheckpoint(
            self.__model_route,
            monitor='val_acc',
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode='auto',
            period=1
        )
    
    def _get_directory_iterator(self, route):
        image_generator = image.ImageDataGenerator(rescale=1.0 / 255)
        
        return image_generator.flow_from_directory(
            directory=route,
            target_size=(self.__width, self.__height),
            batch_size=self.batch_size,
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
