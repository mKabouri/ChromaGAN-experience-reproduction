import config
import Data as Data
import numpy as np
import os

import keras
import tensorflow as tf
from keras.optimizers import SGD
from keras.layers import Input, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.optimizers import Adam

                
    
def createUNetModel(pretrainedWeigths=None):
    inputImg = Input(shape=inImgShape)
        
    downsamplingLayers = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(inputImg)
    downsamplingLayers = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(downsamplingLayers)
    downsamplingLayers = MaxPooling2D(pool_size=(2, 2))(downsamplingLayers)
    downsamplingLayers = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(downsamplingLayers)
    downsamplingLayers = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(downsamplingLayers)
    downsamplingLayers = MaxPooling2D(pool_size=(2, 2))(downsamplingLayers)
    downsamplingLayers = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(downsamplingLayers)
    downsamplingLayers = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(downsamplingLayers)
    upsamplingLayers = UpSampling2D(size=(2, 2), interpolation='nearest')(downsamplingLayers)
    upsamplingLayers = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(upsamplingLayers)
    upsamplingLayers = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(upsamplingLayers)
    upsamplingLayers = UpSampling2D(size=(2, 2), interpolation='nearest')(upsamplingLayers)
    upsamplingLayers = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(upsamplingLayers)
    upsamplingLayers = Conv2D(filters=3, kernel_size=1)(upsamplingLayers)

    model = Model(inputs=inputImg, outputs=upsamplingLayers)
            
    if pretrainedWeigths:
    	model.load_weights(pretrainedWeigths)
            
    return model
    
    

    
if __name__ == '__main__':
    data = Data.Data()
            
    inImgShape = (config.ImageSize, config.ImageSize, 1)
            
    optimizer = Adam(1e-4, 0.5)

    autoencoder = createUNetModel()
    autoencoder.compile(optimizer=optimizer, loss=['mse'], metrics=['accuracy'])

    print("===============Training===============\n")
    
    history = autoencoder.fit(tf.convert_to_tensor(data.X_train), tf.convert_to_tensor(data.y_train), batch_size=config.batchSize
                             , epochs=config.numEpochs, validation_split=0.2)
        

    autoencoder.save("colorization.h5")
        
    