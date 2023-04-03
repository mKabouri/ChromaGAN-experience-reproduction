import numpy as np
import cv2
import os
import config
    

def loadData(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class Data():
    def __init__(self):
        self.labelNames = np.array([loadData(config.labelNamesFile)])
        self.dataset = np.array([loadData(os.path.join(config.dataDir, x)) for x in config.dataFiles])

    def rowToMatrix(self, batchIndex, rowIndex):
        assert batchIndex < 5 and batchIndex >= 0, 'We have only 5 batches'
        assert rowIndex < 10000 and rowIndex >= 0, 'We have 10000 rows (images) per batch'
        return self.dataset[batchIndex][b'data'][rowIndex].reshape(3, 32, 32).T

    def getClass(self, batchIndex, rowIndex):
        assert batchIndex < 5 and batchIndex >= 0, 'We have only 5 batches'
        assert rowIndex < 10000 and rowIndex >= 0, 'We have 10000 rows (images) per batch'
        return self.labelNames[0][b'label_names'][self.dataset[batchIndex][b'labels'][rowIndex]].decode("utf-8")



if __name__ == '__main__':
    
    data = Data()
    
    print(data.dataset[0].keys(), data.dataset[1].keys())
    print(len(data.dataset))
    print(data.dataset[len(data.dataset)-1].keys())
    print(data.labelNames)
    
    # show an image from dataset   
    img = data.rowToMatrix(0, 2)
    cv2.imwrite("./test_show.png", img)
    print(img.shape)

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(data.getClass(0, 2))
    