import config
import os

import model as model
#import Data as Data
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

"""
numChannels = 3 # We want to predict RGB
colorization = model.colorizationNet(numChannels)

loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Accuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.Accuracy(name='test_accuracy')

@tf.function
def train_step(grayImages, colorImages):
  with tf.GradientTape() as tape:
    predictions = colorization(grayImages)
    loss = loss_object(colorImages, predictions)
  gradients = tape.gradient(loss, colorization.trainable_variables)
  optimizer.apply_gradients(zip(gradients, colorization.trainable_variables))

  train_loss(loss)
  train_accuracy(colorImages, predictions)

@tf.function
def test_step(grayImages, colorImages):
  predictions = colorization(grayImages)
  t_loss = loss_object(colorImages, predictions)

  test_loss(t_loss)
  test_accuracy(colorImages, predictions)

@tf.function
def train(train_ds, test_ds, epochs):
    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for grayImages, colorImages in train_ds:
            train_step(grayImages, colorImages)

        for testGrayImages, testColorImages in test_ds:
            test_step(testGrayImages, testColorImages)

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
            f'Test Loss: {test_loss.result()}, '
            f'Test Accuracy: {test_accuracy.result() * 100}'
        )
"""     
###############################################################

if __name__ == '__main__':
    """
    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")
    
    train_ds = tf.data.Dataset.from_tensor_slices(
        (X_train, y_train)).shuffle(10000).batch(config.batchSize)

    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(config.batchSize)

    if tf.test.is_gpu_available():    
        print("===============Training with GPU===============\n")
    else:    
        print("===============Training without GPU===============\n")

    EPOCHS = config.numEpochs
    train(train_ds, test_ds, EPOCHS)
    """
    """
    if config.ImagesFormat == 'Lab' and not(os.path.exists("X_train_Lab.npy")):
        raise ValueError("You have to exexute Data.py first from root to have data files and chose an adequat images format in config.py")
    elif config.ImagesFormat == 'RGB' and not(os.path.exists("X_train.npy")):
        raise ValueError("You have to exexute Data.py first from root to have data files and chose an adequat images format in config.py")
    """
    if config.dataset == 'Div2K':
        numChannels = 3 # We want to predict RGB
    
        # Load data
        X_train = np.load("X_train_Div2K.npy")
        y_train = np.load("y_train_Div2K.npy")
    else:
        if config.ImagesFormat == 'RGB':
            numChannels = 3 # We want to predict RGB
        
            # Load data
            X_train = np.load("X_train.npy")
            y_train = np.load("y_train.npy")
        else: 
            numChannels = 2 # We want to predict ab
        
            # Load data
            X_train = np.load("X_train_Lab.npy")
            y_train = np.load("y_train_Lab.npy")


    colorization = model.colorizationNet(numChannels)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    loss = tf.keras.losses.MeanAbsoluteError()
    metrics = ['mae', 'mse']

    colorization.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    if config.dataset == 'Div2K':
        colorization.build((config.batchSize, config.ImageSize, config.ImageSize, 1))
    else:
        colorization.build((config.batchSize, config.ImageSize, config.ImageSize, 1))

    print(colorization.summary())

    history = colorization.fit(X_train, y_train, batch_size=config.batchSize
                        , epochs=config.numEpochs, validation_split=0.2)
                        #, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=2)])

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    if config.ImagesFormat == 'Lab':
        plt.title('Lab prediction loss')
    else:
        plt.title('RGB prediction loss')

    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    if config.dataset == 'Div2K':
        plt.savefig('Div2K_loss.png') 
    else:
        plt.savefig('CIFAR10_loss' + config.ImagesFormat + '.png')
    plt.show()

    if config.dataset == 'Div2K':
        X_test = np.load("X_test_Div2K.npy")
        y_test = np.load("y_test_Div2K.npy")
    else:
        if config.ImagesFormat == 'RGB':
            X_test = np.load("X_test.npy")
            y_test = np.load("y_test.npy")
        else:
            X_test = np.load("X_test_Lab.npy")
            y_test = np.load("X_test_Lab.npy")

    test_scores = colorization.evaluate(X_test, y_test, verbose=2)
    print("Test loss: ", test_scores[0])
    print("Test mae: ", test_scores[1])
    print("Test mse: ", test_scores[2])

    if config.dataset == 'Div2K':
        colorization.save_weights(os.path.join(config.saveModelDir, 'Div2KWithSkipRGBWeights.h5'), save_format='h5')
    else:
        if config.ImagesFormat == 'RGB':
            colorization.save_weights(os.path.join(config.saveModelDir, 'colorizationWithSkipRGBWeights.h5'), save_format='h5')
        else: 
            colorization.save_weights(os.path.join(config.saveModelDir, 'colorizationWithSkipLabWeights.h5'), save_format='h5')