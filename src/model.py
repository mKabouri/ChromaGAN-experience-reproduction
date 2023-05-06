import numpy as np
import config

import tensorflow as tf
from tensorflow.keras.layers import MaxPooling2D, Conv2D, UpSampling2D, BatchNormalization, Concatenate

class convBlock(tf.keras.layers.Layer):
    def __init__(self, numFilters):
        super().__init__()
        self.conv1 = Conv2D(filters=numFilters, kernel_size=(3, 3), padding='same')
        self.batchNorm1 = BatchNormalization()
        self.conv2 = Conv2D(filters=numFilters, kernel_size=(3, 3), activation='relu', padding='same')
        self.batchNorm2 = BatchNormalization()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        return self.batchNorm2(x)

class colorizationNet(tf.keras.Model):
    def __init__(self, numChannels):
        super(colorizationNet, self).__init__()
        self.convDown32 = convBlock(32)
        self.maxPool1 = MaxPooling2D(pool_size=(2, 2))

        self.convDown64 = convBlock(64)
        self.maxPool2 = MaxPooling2D(pool_size=(2, 2))

        self.convDown128 = convBlock(128)
        self.upSampl1 = UpSampling2D(size=(2, 2), interpolation='nearest')

        self.convUp128 = convBlock(128)
        self.upSampl2 = UpSampling2D(size=(2, 2), interpolation='nearest')

        self.convUp64 = convBlock(64)
        self.convUp32 = convBlock(32)

        self.convForSkip1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')
        self.convForSkip2 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')
        self.conv3 = Conv2D(filters=numChannels, kernel_size=(3, 3), activation='relu', padding='same')
        
    def call(self, inputs):
        # Encoder
        convDown32 = self.convDown32(inputs)
        convForSkip1 = self.convForSkip1(convDown32)
        maxPool1 = self.maxPool1(convForSkip1)
        
        convDown64 = self.convDown64(maxPool1)
        convForSkip2 = self.convForSkip2(convDown64)
        maxPool2 = self.maxPool2(convForSkip2)
        
        convDown128 = self.convDown128(maxPool2)
        
        # Decoder
        convUp128 = self.convUp128(convDown128)
        upSampl1 = self.upSampl1(convUp128)
        
        skipConn1 = Concatenate()([convForSkip2, upSampl1])
        
        convUp64 = self.convUp64(skipConn1)
        upSampl2 = self.upSampl2(convUp64)
        
        skipConn2 = Concatenate()([convForSkip1, upSampl2])
        
        convUp32 = self.convUp32(skipConn2)
        return self.conv3(convUp32)
