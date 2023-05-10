import os
import numpy as np

# Directory information
dataset = "CIFAR-10"
rootDir = os.path.abspath('./')
dataDir = os.path.join(rootDir, 'dataset/')
Sourcedir = os.path.join(rootDir, 'src/')
saveModelDir = os.path.join(rootDir, 'saveModel/')
dataFiles = np.sort([x for x in os.listdir(dataDir) if x.startswith('data') or x.startswith('test')])
labelNamesFile = os.path.join(dataDir, 'batches.meta')
saveRGBSamples = os.path.join(rootDir, 'samples/rgbFormat/result/')
saveLabSamples = os.path.join(rootDir, 'samples/LabSpace/result/')

ImagesFormat = 'Lab' # You can chose between Lab or RGB. If you want 
                     # to save Data in form of .npy you can decomment 
                     # line 82 in Data.py or you can use Data class directly

batchSize = 10
numEpochs = 6

ImageSize = 32
