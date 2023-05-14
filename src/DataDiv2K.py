import os
import config

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray, rgb2lab, lab2rgb
from skimage.transform import resize

data_dir = 'dataset/div2k/'
image_paths = os.listdir(data_dir)

train_paths = image_paths[:int(0.8 * len(image_paths))]
test_paths = image_paths[int(0.8 * len(image_paths)):]

def readImage(image_path):
    if not(os.path.isdir(os.path.join(data_dir, image_path))):
        image = io.imread(os.path.join(data_dir, image_path))
        return resize(image, (128, 128))

def toGray(img):
    return rgb2gray(img).reshape(128, 128, 1)


y_train = np.array([readImage(image_path) for image_path in train_paths if not(os.path.isdir(image_path))])
X_train = np.array([toGray(x) for x in y_train])

y_test = np.array([readImage(image_path) for image_path in test_paths if not(os.path.isdir(image_path))])
X_test = np.array([toGray(x) for x in y_test])

file_X_train = "X_train_Div2K.npy"
file_y_train = "y_train_Div2K.npy"
file_X_test = "X_test_Div2K.npy"
file_y_test = "y_test_Div2K.npy"     
            
with open(file_y_train, 'wb') as f:
    np.save(f, y_train)
with open(file_X_train, 'wb') as f:
    np.save(f, X_train)  
with open(file_y_test, 'wb') as f:
    np.save(f, y_test)
with open(file_X_test, 'wb') as f:
    np.save(f, X_test) 
