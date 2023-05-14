import os
import numpy as np

# Directory information
dataset = "cifar-10" #There is also cifar-10 dataset
rootDir = os.path.abspath('./')
dataDir = os.path.join(rootDir, 'dataset/' + dataset + '/')
Sourcedir = os.path.join(rootDir, 'src/')
saveModelDir = os.path.join(rootDir, 'saveModel/')

if dataset == 'cifar-10':
    dataFiles = np.sort([x for x in os.listdir(dataDir) if x.startswith('data') or x.startswith('test')])
    labelNamesFile = os.path.join(dataDir, 'batches.meta')

saveRGBSamples = os.path.join(rootDir, 'samples/rgbFormat/result/')
saveLabSamples = os.path.join(rootDir, 'samples/LabSpace/result/')

ImagesFormat = 'Lab' # You can chose between Lab or RGB. If you want 
                     # to save Data in form of .npy you can decomment 
                     # line 82 in Data.py or you can use Data class directly



batchSize = 10
if dataset == 'Div2K':
    numEpochs = 25
else:
    numEpochs = 10

if dataset == 'Div2K':
    ImageSize = 128
else:
    ImageSize = 32