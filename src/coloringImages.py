import model as model
import os
import config

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray, rgb2lab, lab2rgb


def showImage(self, img):
        plt.imshow(img)
        plt.show()

def denormalize_lab(lab_norm):
    L_norm, a_norm, b_norm = lab_norm[:,:,0], lab_norm[:,:,1], lab_norm[:,:,2]
    L = L_norm * 100.0
    a = a_norm * 127.0
    b = b_norm * 127.0
    image_lab = np.stack([L, a, b], axis=2)
    return image_lab

def backToRGBPipeline(gray, ab):
    return lab2rgb(denormalize_lab(np.concatenate((gray, ab), axis=2)))

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
            res = np.concatenate((np.concatenate((backToRGBPipeline(X_test[i], y_test[i+n]), np.stack((batchX[i].reshape(32, 32),)*3, axis=-1))), backToRGBPipeline(batchX[i], colorImages[i])))
        else:
            res = np.concatenate((np.concatenate((y_test[i+n], np.stack((batchX[i].reshape(32, 32),)*3, axis=-1))), colorImages[i].reshape(32, 32, 3)))
        cv2.imwrite(dirToSave + "new_arch_image_" + str(n + i) + ".png", res)
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
