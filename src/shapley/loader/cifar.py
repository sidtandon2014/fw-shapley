import copy
import numpy as np
import pickle
from sklearn.utils import shuffle
from tensorflow import keras

from shapley.loader import Loader

class CIFAR(Loader):
    def __init__(self, num_train,num_test=None,all_classes=False, seed=42):
        self.name = 'cifar'
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
        cifar10 = keras.datasets.cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        x_train = x_train.transpose(0,3,1,2)
        x_train = x_train.astype(np.float32) / 255
        x_test = x_test.transpose(0,3,1,2)
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