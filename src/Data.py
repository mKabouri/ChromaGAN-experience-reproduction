import numpy as np
import cv2
import os
import config


def loadData(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def deprocess(imgs):
    imgs = imgs * 255
    imgs[imgs > 255] = 255
    imgs[imgs < 0] = 0
    return imgs.astype(np.uint8)

def concatenateLabToRGB(gray, ab):
    concat = np.concatenate((gray, ab), axis=2)
    return cv2.cvtColor(concat, cv2.COLOR_Lab2RGB)


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
        if self.imagesFormat == 'Lab':
            return  img[:,:,0]
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    def colorImg(self, img):
        img = img/255
        img = img.astype("float32")
        if self.imagesFormat == 'Lab':
            return cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        return img

    def __getLearningData(self):
        trainImages = np.array([self.colorImg(np.moveaxis(x.reshape(3, 32, 32), 0, -1)) for t in self.dataset[:len(self.dataset) - 1] for x in t[b'data']])
        testImages = np.array([self.colorImg(np.moveaxis(x.reshape(3, 32, 32), 0, -1)) for x in self.dataset[len(self.dataset)-1][b'data']])
        y_train = np.array([y[:, :, 1:] for y in trainImages])
        y_test = np.array([y[:, :, 1:] for y in testImages])  

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


if __name__ == '__main__':
    
    data = Data(config.ImagesFormat)
    
    #data.save()
    
    print("Ok")  
    
