import os
import config

import matplotlib.pyplot as plt
from skimage.color import rgb2gray, rgb2lab, lab2rgb
import numpy as np

def loadData(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def normalize_lab(lab):
    L, a, b = lab[:,:,0], lab[:,:,1], lab[:,:,2]
    L_norm = L / 100.0
    a_norm = a / 127.0
    b_norm = b / 127.0
    image_lab_norm = np.stack([L_norm, a_norm, b_norm], axis=2)
    return image_lab_norm

def denormalize(img):
    if config.ImagesFormat == 'Lab':
        L_norm, a_norm, b_norm = img[:,:,0], img[:,:,1], img[:,:,2]
        L = L_norm * 100.0
        a = a_norm * 127.0
        b = b_norm * 127.0
        image_lab = np.stack([L, a, b], axis=2)
        return image_lab
    else:     
        img = img * 255
        img[img > 255] = 255
        img[img < 0] = 0
        return img.astype(np.uint8)

def backToRGB(gray, ab):
    return lab2rgb(denormalize(np.concatenate((gray, ab), axis=2)))


class Data():
    def __init__(self, imagesFormat):
        assert imagesFormat == 'Lab' or imagesFormat == 'RGB', 'You have to precise the images format "Lab" or "RGB"'
        self.imagesFormat = imagesFormat
        self.labelNames = np.array([loadData(config.labelNamesFile)])
        self.dataset = np.array([loadData(os.path.join(config.dataDir, x)) for x in config.dataFiles])
        self.X_train, self.y_train, self.X_test, self.y_test = self.__getLearningData()

    def rowToMatrix(self, batchIndex, rowIndex):
        assert batchIndex < 5 and batchIndex >= 0, 'We have only 5 batches'
        assert rowIndex < 10000 and rowIndex >= 0, 'We have 10000 rows (images) per batch'
        return np.moveaxis(self.dataset[batchIndex][b'data'][rowIndex].reshape(3, 32, 32), 0, -1)

    def getClass(self, batchIndex, rowIndex):
        assert batchIndex < 5 and batchIndex >= 0, 'We have only 5 batches'
        assert rowIndex < 10000 and rowIndex >= 0, 'We have 10000 rows (images) per batch'
        return self.labelNames[0][b'label_names'][self.dataset[batchIndex][b'labels'][rowIndex]].decode("utf-8")
    
    def toGray(self, img):
        return rgb2gray(img)

    def __getLearningData(self):
        trainImages = np.array([np.moveaxis(x.reshape(3, 32, 32), 0, -1) for t in self.dataset[:len(self.dataset) - 1] for x in t[b'data']])
        testImages = np.array([np.moveaxis(x.reshape(3, 32, 32), 0, -1) for x in self.dataset[len(self.dataset)-1][b'data']])
        
        if config.ImagesFormat == 'Lab':
            y_train = np.array([normalize_lab(rgb2lab(y))[:, :, 1:] for y in trainImages])
            y_test = np.array([normalize_lab(rgb2lab(y))[:, :, 1:] for y in testImages])
            X_train = np.array([self.toGray(y).reshape(32, 32, 1) for y in trainImages])
            X_test = np.array([self.toGray(y).reshape(32, 32, 1) for y in testImages])
        else:
            y_train = trainImages.copy()/255
            y_test = testImages.copy()/255
            X_train = np.array([self.toGray(y).reshape(32, 32, 1) for y in trainImages])
            X_test = np.array([self.toGray(y).reshape(32, 32, 1) for y in testImages])

        return X_train, y_train, X_test, y_test
        
    def save(self):
        if self.imagesFormat == 'RGB':
            file_X_train = "X_train.npy"
            file_y_train = "y_train.npy"
            file_X_test = "X_test.npy"
            file_y_test = "y_test.npy"
        else:
            file_X_train = "X_train_Lab.npy"
            file_y_train = "y_train_Lab.npy"
            file_X_test = "X_test_Lab.npy"
            file_y_test = "y_test_Lab.npy"     
            
        with open(file_X_train, 'wb') as f:
            np.save(f, self.X_train)  
        with open(file_y_train, 'wb') as f:
            np.save(f, self.y_train)
        with open(file_X_test, 'wb') as f:
            np.save(f, self.X_test)  
        with open(file_y_test, 'wb') as f:
            np.save(f, self.y_test)
    
        print("Data (", config.ImagesFormat, ") is successfully saved in npy format at: ", config.rootDir)

    def showImage(self, img):
        plt.imshow(img)
        plt.show()


    
if __name__ == '__main__':
    
    data = Data(config.ImagesFormat)
    
    print(data.X_train.shape, data.y_train.shape)

    data.save()
    """
    if config.ImagesFormat == 'Lab':
        L = data.X_train[0]
        print(L)
        ab = data.y_train[0]
        data.showImage(backToRGB(L, ab))
    else:
        img = data.X_train[6]
        img_test = data.y_train[6]
        print(img)
        data.showImage(denormalize(img))
        print(denormalize(img_test))
        data.showImage(denormalize(img_test))
        print(denormalize(img_test))
    """
    print("Ok")  
    

