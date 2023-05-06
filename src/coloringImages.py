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

def concatenateGrayWithAB(gray, ab):
    a = ab[:, :, 0].astype('float32')
    b = ab[:, :, 1].astype('float32')
    a = a.reshape(32, 32, 1)
    b = b.reshape(32, 32, 1)
    gray = gray.astype('float32')
    concat = cv2.merge((gray, a, b)).astype('float32')
    return cv2.cvtColor(concat, cv2.COLOR_Lab2RGB)

def coloringSamples(colorization, nbSamples, dirToSave, pathToX_test, pathToy_test, weightsFileName):
    X_test = np.load(pathToX_test)
    y_test = np.load(pathToy_test)
    n = np.random.randint(0, len(X_test))
    batchX = []
    if n + nbSamples > len(X_test):
        batchX = X_test[n - nbSamples: n]
    else:
        batchX = X_test[n : n + nbSamples]
    colorization.load_weights(os.path.join(config.saveModelDir, weightsFileName)) 
    colorImages = np.array([colorization.predict(x.reshape(1, 32, 32, 1)) for x in batchX])
    for i in range(nbSamples):
        if config.ImagesFormat == 'Lab':
            res = np.concatenate((np.concatenate((concatenateGrayWithAB(X_test[i], y_test[i+n]), np.stack((batchX[i].reshape(32, 32),)*3, axis=-1))), concatenateGrayWithAB(batchX[i], colorImages[i].reshape(32, 32, 2))))
        else:
            res = np.concatenate((np.concatenate((y_test[i+n], np.stack((batchX[i].reshape(32, 32),)*3, axis=-1))), colorImages[i].reshape(32, 32, 3)))
        cv2.imwrite(dirToSave + "new_arch_image_" + str(n + i) + ".png", deprocess(res))
    print("Colored and saved at: ", dirToSave)
    return
    
if __name__ == '__main__':
    
    if config.ImagesFormat == 'Lab':
        numChannels = 2
        weightsFileName = 'colorizationWithSkipLabWeights.h5'
        pathToX_test = 'X_test_Lab.npy'
        pathToy_test = 'y_test_Lab.npy'
        dirToSave = config.saveLabSamples
    else:
        numChannels = 3
        weightsFileName = 'colorizationWithSkipRGBWeights.h5'
        pathToX_test = 'X_test.npy'
        pathToy_test = 'y_test.npy'
        dirToSave = config.saveRGBSamples
    
    nbSamples = 5
    
    
    if not(os.path.exists(pathToX_test)):
        raise ValueError("You have to exexute Data.py first from root to have data files")
    
    colorizationModel = model.colorizationNet(numChannels)
    colorizationModel.build((1, 32, 32, 1))
        
    coloringSamples(colorizationModel, nbSamples, dirToSave, pathToX_test, pathToy_test, weightsFileName)
