import os
import numpy as np

# Directory information
dataset = "CIFAR-10"
rootDir = os.path.abspath('./')
dataDir = os.path.join(rootDir, 'dataset/')
dataFiles = np.sort([x for x in os.listdir(dataDir) if x.startswith('data') or x.startswith('test')])
labelNamesFile = os.path.join(dataDir, 'batches.meta')

batchSize = 10
numEpochs = 10

ImageSize = 32
