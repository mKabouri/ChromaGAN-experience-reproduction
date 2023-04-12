import config
import Data as Data
import numpy as np
import os

import keras
from keras.optimizers import SGD
from keras.layers import Input, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.optimizers import Adam

class model():
    def __init__(self):
        self.data = Data.Data()
        
        self.inImg = (config.ImageSize, config.ImageSize, 1)
        
        optimizer = Adam(1e-4, 0.5)

        self.autoencoder = model()
        self.autoencoder.compile(optimizer=optimizer, loss=['mse', 'kld'], metrics=['accuracy'])
        
        self.trained = 0
        
    
    def model(self, pretrainedWeigths=None):
        inputImg = Input(shape=self.inImg)
        
        downsamplingLayers = Conv2D(filters=32, kernel_size=(3, 3), activation='ReLu', padding='same')(input)
        downsamplingLayers = Conv2D(filters=32, kernel_size=(3, 3), activation='ReLu', padding='same')(downsamplingLayers)
        downsamplingLayers = MaxPooling2D(pool_size=(2, 2), strides=1)(downsamplingLayers)
        downsamplingLayers = Conv2D(filters=32, kernel_size=(3, 3), activation='ReLu', padding='same')(downsamplingLayers)
        downsamplingLayers = Conv2D(filters=64, kernel_size=(3, 3), activation='ReLu', padding='same')(downsamplingLayers)
        downsamplingLayers = MaxPooling2D(pool_size=(2, 2), strides=1)(downsamplingLayers)
        downsamplingLayers = Conv2D(filters=64, kernel_size=(3, 3), activation='ReLu', padding='same')(downsamplingLayers)
        downsamplingLayers = Conv2D(filters=128, kernel_size=(3, 3), activation='ReLu', padding='same')(downsamplingLayers)

        upsamplingLayers = Conv2DTranspose(filters=128, kernel_size=(3, 3), activation='ReLu')(downsamplingLayers)
        upsamplingLayers = Conv2D(filters=128, kernel_size=(3, 3), activation='ReLu', padding='same')(upsamplingLayers)
        upsamplingLayers = Conv2D(filters=64, kernel_size=(3, 3), activation='ReLu', padding='same')(upsamplingLayers)
        upsamplingLayers = Conv2DTranspose(filters=64, kernel_size=(3, 3), activation='ReLu')(upsamplingLayers)
        upsamplingLayers = Conv2D(filters=64, kernel_size=(3, 3), activation='ReLu', padding='same')(upsamplingLayers)
        upsamplingLayers = Conv2D(filters=32, kernel_size=(3, 3), activation='ReLu', padding='same')(upsamplingLayers)

        model = Model(inputs=[inputImg], outputs=[downsamplingLayers])
            
        if pretrainedWeigths:
    	    model.load_weights(pretrainedWeigths)
            
        return model
    
    def train(self):
        model_checkpoint = ModelCheckpoint('checkPointsTraining.hdf5', monitor='val_accuracy'
                                           , mode='max', save_best_only=True)
        self.autoencoder.fit([self.data.X_train], [self.data.y_train], batch_size=config.batchSize
                             , epochs=config.numEpochs, validation_split=0.2, callbacks=[model_checkpoint])
        self.trained = 1
        
    def testModel(self):
        pass        
    

    
if __name__ == '__main__':
    colorization = model()
    
    if not colorization.trained:
        print("===============Training===============\n")
        colorization.train()
            
    