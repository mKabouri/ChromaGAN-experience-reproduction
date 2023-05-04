import cv2
import numpy as np

import model as model
import os
import config
import matplotlib.pyplot as plt

def deprocess(imgs):
    imgs = imgs * 255
    imgs[imgs > 255] = 255
    imgs[imgs < 0] = 0
    return imgs.astype(np.uint8)

def coloringSamples(colorization, nbSamples):
    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")
    n = np.random.randint(0, len(X_test))
    batchX = []
    if n + nbSamples > len(X_test):
        batchX = X_test[n - nbSamples: n]
    else:
        batchX = X_test[n : n + nbSamples]
    colorization.load_weights(os.path.join(config.saveModelDir, 'colorizationRGBWeights.h5')) 
    colorImages = np.array([colorization.predict(x.reshape(1, 32, 32, 3)) for x in batchX])
    print(colorImages.shape)
    print(batchX.shape)
    for i in range(nbSamples):
        res = np.concatenate((np.concatenate((colorImages[i].reshape(32, 32, 3), batchX[i])), y_test[i+n]))
        cv2.imwrite(config.saveRGBSamples + "Image_" + str(n + i) + ".png", deprocess(res))
    print("Colored and saved at: ", config.saveRGBSamples)
    return
    
if __name__ == '__main__':
    colorizationModel = model.colorizationNet(3)
    colorizationModel.build((1,32,32,3))
    coloringSamples(colorizationModel, 5)
