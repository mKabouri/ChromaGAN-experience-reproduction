import numpy as np
import config

import tensorflow as tf
from tensorflow.keras.layers import MaxPooling2D, Conv2D, UpSampling2D, BatchNormalization, Conv2DTranspose

class downSamplingLayer(tf.keras.layers.Layer):
    def __init__(self, numFilters):
        super().__init__()
        self.conv1 = Conv2D(filters=numFilters, kernel_size=(3, 3), activation='relu', padding='same')
        self.batchNorm1 = BatchNormalization()
        self.conv2 = Conv2D(filters=numFilters, kernel_size=(3, 3), activation='relu', padding='same')
        self.batchNorm2 = BatchNormalization()
        self.maxPool = MaxPooling2D(pool_size=(2, 2))

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = self.batchNorm2(x)
        return self.maxPool(x)

class upSamplingLayer(tf.keras.layers.Layer):
    def __init__(self, numFilters):
        super().__init__()
        self.conv1 = Conv2D(filters=numFilters, kernel_size=(3, 3), activation='relu', padding='same')
        self.batchNorm1 = BatchNormalization()
        self.conv2 = Conv2D(filters=numFilters, kernel_size=(3, 3), activation='relu', padding='same')
        self.batchNorm2 = BatchNormalization()
        self.upSampling = UpSampling2D(size=(2, 2), interpolation='nearest')
        
    def call(self, inputs):
        x = self.upSampling(inputs)
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        return self.batchNorm2(x)

class colorizationNet(tf.keras.Model):
    def __init__(self, numChannels):
        super(colorizationNet, self).__init__()
        self.downSampl32 = downSamplingLayer(32)
        self.downSampl64 = downSamplingLayer(64)
        self.downSampl128 = downSamplingLayer(128)
        self.upSampl128 = upSamplingLayer(128)
        self.upSampl64 = upSamplingLayer(64)
        self.upSampl32 = upSamplingLayer(32)
        self.conv3 = Conv2D(filters=numChannels, kernel_size=(3, 3), activation='relu', padding='same')

    def call(self, inputs):
        x = self.downSampl32(inputs)
        x = self.downSampl64(x)
        x = self.downSampl128(x)
        x = self.upSampl128(x)
        x = self.upSampl64(x)
        x = self.upSampl32(x)
        return self.conv3(x)
