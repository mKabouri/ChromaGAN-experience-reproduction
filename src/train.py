import model as model
import Data as Data
import config

import tensorflow as tf

numChannels = 3 # We use RGB
colorization = model.colorizationNet(numChannels)


"""
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
"""
###############################################################

if __name__ == '__main__':
    data = Data.Data()

    """
    train_ds = tf.data.Dataset.from_tensor_slices(
        (data.X_train, data.y_train)).shuffle(10000).batch(config.batchSize)

    print("## Here ## ", train_ds)

    test_ds = tf.data.Dataset.from_tensor_slices((data.X_test, data.y_test)).batch(config.batchSize)

    if tf.test.is_gpu_available():    
        print("===============Training with GPU===============\n")
    else:    
        print("===============Training without GPU===============\n")

    # Training loop
    EPOCHS = config.numEpochs

    for epoch in range(EPOCHS):
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
    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.MeanSquaredError()
    metrics = ['accuracy']

    colorization.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    history = colorization.fit(data.X_train, data.y_train, batch_size=config.batchSize
                        , epochs=config.numEpochs, validation_split=0.2
                        , callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)])

    test_scores = colorization.evaluate(data.X_test, data.y_test, verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])
    
    colorization.save("colorization.h5")

