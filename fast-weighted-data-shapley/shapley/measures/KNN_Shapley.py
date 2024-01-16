import numpy as np

import torch
import torch.nn.functional as F

from shapley.measures import Measure

class KNN_Shapley(Measure):

    def __init__(self, K=10):
        self.name = 'KNN_Shapley'
        self.K = K


    def _get_shapley_value_torch(self, X_train, y_train, X_test, y_test):
        '''
        Input: X_train and X_test refer to the input features for the KNN classifier (these features could be output of some feature extractor)
               y_train and y_test are labels
        Output: (N_train, 1) > importance of each training sample w.r.t. prediction of the model on the test set (X_test, y_test)
        '''
        N = len(X_train)
        M = len(X_test)

        # uncomment below line for orgiinal code
        # dist = torch.cdist(X_train.view(len(X_train), -1), X_test.view(len(X_test), -1))
        dist = torch.cdist(X_train.reshape(len(X_train), -1), X_test.reshape(len(X_test), -1))
        _, indices = torch.sort(dist, axis=0) # sort the values in each of the columns independently
        y_sorted = y_train[indices]

        score = torch.zeros_like(dist)

        score[indices[N-1], range(M)] = (y_sorted[N-1] == y_test).float() / N
        for i in range(N-2, -1, -1):
            score[indices[i], range(M)] = score[indices[i+1], range(M)] + \
                                        1/self.K * ((y_sorted[i] == y_test).float() - (y_sorted[i+1] == y_test).float()) * min(self.K, i+1) / (i+1)
        return score.mean(axis=1)


    def _get_shapley_value_np(self, X_train, y_train, X_test, y_test):
        '''
        Input: X_train and X_test refer to the input features for the KNN classifier (these features could be output of some feature extractor as well)
               y_train and y_test are labels
        '''
        # print('KNN_shapley.py (get_shapley_value_np, train) > ',X_train==None)
        # print('KNN_shapley.py (get_shapley_value_np, test) > ',X_test==None)
        N = len(X_train)
        M = len(X_test)
        s = np.zeros((N, M))

        for i, (X, y) in enumerate(zip(X_test, y_test)):
            diff = (X_train - X).reshape(N, -1)
            dist = np.einsum('ij, ij->i', diff, diff)
            idx = np.argsort(dist)
            ans = y_train[idx]
            s[idx[N - 1]][i] = float(ans[N - 1] == y) / N
            cur = N - 2
            for j in range(N - 1):
                s[idx[cur]][i] = s[idx[cur + 1]][i] + float(int(ans[cur] == y) - int(ans[cur + 1] == y)) / self.K * (min(cur, self.K - 1) + 1) / (cur + 1)
                cur -= 1 
        return np.mean(s, axis=1)


    def score(self, X_train, y_train, X_test, y_test, model_family='', model=None):
        self.model = model

        if model_family == "ResNet":
            resnet = model
            # X_test = X_test.reshape(X_test.shape[0],1,28,28)
            # X_test = np.concatenate((X_test,X_test,X_test),axis=1)
            X_out1 = resnet.layer1(F.relu(resnet.bn1(resnet.conv1(torch.from_numpy(X_train)))))
            X_test_out1 = resnet.layer1(F.relu(resnet.bn1(resnet.conv1(torch.from_numpy(X_test))))) # 64, 32, 32
            X_out2 = resnet.layer2(X_out1)
            X_test_out2 = resnet.layer2(X_test_out1) # 64, 32, 32
            X_out3 = resnet.layer3(X_out2)
            X_test_out3 = resnet.layer3(X_test_out2) # 64, 32, 32

            return self._get_shapley_value_np(X_train=X_out1, y_train=y_train, X_test=X_test_out1, y_test=y_test)

        if model_family == "NN":
            nn = model
            X_feature = np.maximum(np.matmul(X_train, nn.coefs_[0]) + nn.intercepts_[0], 0) 
            #X_feature = ACTIVATIONS['relu'](np.matmul(X_train, nn.coefs_[0]) + nn.intercepts_[0]) # ACTIVATIONS['relu'] was giving None as output
            # X_test_feature = ACTIVATIONS['relu'](np.matmul(X_test, nn.coefs_[0]) + nn.intercepts_[0]) # ACTIVATIONS['relu'] was giving None as output
            X_test_feature = np.maximum(np.matmul(X_test, nn.coefs_[0]) + nn.intercepts_[0], 0)

            return self._get_shapley_value_np(X_train=X_feature, y_train=y_train, X_test=X_test_feature, y_test=y_test)

        return self._get_shapley_value_np(X_train, y_train, X_test, y_test)