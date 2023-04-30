import numpy as np
import cv2
import os
import config

""" 
Add a description of data dormat
"""


def loadData(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class Data():
    def __init__(self):
        self.labelNames = np.array([loadData(config.labelNamesFile)])
        self.dataset = np.array([loadData(os.path.join(config.dataDir, x)) for x in config.dataFiles])
        self.X_train, self.y_train, self.X_test, self.y_test = self.__getLearningData()

    def rowToMatrix(self, batchIndex, rowIndex):
        assert batchIndex < 5 and batchIndex >= 0, 'We have only 5 batches'
        assert rowIndex < 10000 and rowIndex >= 0, 'We have 10000 rows (images) per batch'
        return self.dataset[batchIndex][b'data'][rowIndex].reshape(3, 32, 32).T

    def getClass(self, batchIndex, rowIndex):
        assert batchIndex < 5 and batchIndex >= 0, 'We have only 5 batches'
        assert rowIndex < 10000 and rowIndex >= 0, 'We have 10000 rows (images) per batch'
        return self.labelNames[0][b'label_names'][self.dataset[batchIndex][b'labels'][rowIndex]].decode("utf-8")

    def toGray(self, img):
        return np.stack((cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),)*3, axis=-1)
    
    def __getLearningData(self):
        y_train = np.array([x.reshape(3, 32, 32).T for t in self.dataset[:len(self.dataset) - 1] for x in t[b'data']])
        y_test = np.array([x.reshape(3, 32, 32).T for x in self.dataset[len(self.dataset)-1][b'data']])
        X_train = np.array([self.toGray(y).reshape(32, 32, 3) for y in y_train])
        X_test = np.array([self.toGray(y).reshape(32, 32, 3) for y in y_test])
        return X_train/255, y_train/255, X_test/255, y_test/255
    
"""
with open(file_X, 'wb') as f:
    np.save(f, X)  
with open(file_Y, 'wb') as f:
    np.save(f, Y)
"""
if __name__ == '__main__':
    
    data = Data()

    print(data.dataset[0].keys(), data.dataset[1].keys())
    print(data.X_train.shape, data.X_test.shape, data.y_train.shape, data.y_test.shape)
    print(type(data.X_train), type(data.X_test), type(data.y_train), type(data.y_test))
   
    
    file_X_train = "X_train.npy"
    file_y_train = "y_train.npy"
    file_X_test = "X_test.npy"
    file_y_test = "y_test.npy"
    
    with open(file_X_train, 'wb') as f:
        np.save(f, data.X_train)  
    with open(file_y_train, 'wb') as f:
        np.save(f, data.y_train)
    with open(file_X_test, 'wb') as f:
        np.save(f, data.X_test)  
    with open(file_y_test, 'wb') as f:
        np.save(f, data.y_test)
    
    print(data.dataset[len(data.dataset)-1].keys())
    print(data.labelNames)
    """
    # show an image from dataset   
    img = data.rowToMatrix(0, 2)

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(data.getClass(0, 2))
    
    
    gray_img = data.toGray(img)
    
    cv2.imshow('image 2', gray_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("color image shape = ", img.shape)
    print("gray image shape = ", gray_img.shape)
    
    print("Ok")
    """    
    """
    # Show from X_train and y_train
    print(data.X_train[2].shape)
    print(data.y_train[2].shape)
    
    cv2.imshow('image X_train', data.X_train[2])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imshow('image y_train', data.y_train[2])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    """
    # With resizing and resclaling
    #result = resize_and_rescale(img)
    #print(type(result))
    
    cv2.imshow('image after sequential', result.numpy())
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    print("Ok")  
    
