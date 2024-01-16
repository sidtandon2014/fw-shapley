import copy
import numpy as np
from sklearn.utils import shuffle
from tensorflow import keras

from shapley.loader import Loader

class FashionMnist(Loader):

    def __init__(self, num_train,num_test=None,all_classes=False, seed=42):
        self.name = 'fashion_mnist'
        self.all_classes=all_classes

        self.x_train, self.y_train, self.x_test, self.y_test = self.load_data()

        # return slices of train and test
        if num_train is None: self.num_train = self.x_train.shape[0]
        else:  self.num_train = num_train
        if num_test is None: self.num_test = self.num_train // 10
        else: self.num_test = num_test
        
        self.shuffle_data(seed)
        
        self.x_train = self.x_train[:self.num_train]
        self.y_train = self.y_train[:self.num_train]
        self.x_test = self.x_test[:self.num_test]
        self.y_test = self.y_test[:self.num_test]

    def load_data(self):
        fashion_mnist = keras.datasets.fashion_mnist
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

        if self.all_classes==False: # only uses two classes for training
            indice_0 = np.where(y_train==0)[0]
            indice_1 = np.where(y_train==6)[0]
            indice_all = np.hstack((indice_0, indice_1))
            x_train = x_train[indice_all]
            y_train = np.hstack((np.zeros(len(indice_0), dtype=np.int64), np.ones(len(indice_1), dtype=np.int64)))
            indice_0 = np.where(y_test==0)[0]
            indice_1 = np.where(y_test==6)[0]
            indice_all = np.hstack((indice_0, indice_1))
            x_test = x_test[indice_all]
            y_test = np.hstack((np.zeros(len(indice_0), dtype=np.int64), np.ones(len(indice_1), dtype=np.int64)))

        x_train = x_train[:,np.newaxis,:,:]
        x_train = x_train.astype(np.float32) / 255
        x_test = x_test[:,np.newaxis,:,:] # np.reshape(x_test, [-1, 28 * 28])
        x_test = x_test.astype(np.float32) / 255

        return x_train, y_train, x_test, y_test


    def shuffle_data(self,seed):
        np.random.seed(seed) # this will ensure the shuffle is same every time so that the results are reproducible
        ind = np.random.permutation(len(self.x_train))
        self.x_train, self.y_train = self.x_train[ind], self.y_train[ind]
        ind = np.random.permutation(len(self.x_test))
        self.x_test, self.y_test = self.x_test[ind], self.y_test[ind]

    def prepare_data(self):
        return self.x_train, self.y_train, self.x_test, self.y_test