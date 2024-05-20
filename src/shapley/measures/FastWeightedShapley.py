import numpy as np

import torch
import torch.nn.functional as F
import sys
import os

from shapley.measures import Measure
sys.path.append('../')
# from models.pytorch_fitmodule.utils import get_loader
from datasets import load_FashionMNIST, get_split_dataset, extract_features
from train import ParameterizedShapleyEstimator
from networks.model import EstimatorNetwork
from datasets import get_split_dataset, extract_features
from torch.utils.data import DataLoader, TensorDataset

# device = "cpu"#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FastWeightedShapley(Measure):

    def __init__(self, K=10, model_checkpoint_dir = "./checkpoints", alpha= 16, beta=1):
        self.name = 'FastWeightedShapley'
        self.K = K
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model_family=model_family
        self.model_checkpoint_dir=model_checkpoint_dir
        self.model_path = f"{model_checkpoint_dir}/explainer_model_final.pth"
        self.alpha=alpha
        self.beta = beta

    def set(self, **kwargs):
        if "y_flipped" in kwargs:
            self.y_flipped = kwargs["y_flipped"]

    

    def _get_shapley_value_torch(self, X_train, y_train, X_test, y_test, test_inf_batch_size=1024):
        '''
        Input: X_train and X_test refer to the input features for the KNN classifier (these features could be output of some feature extractor)
               y_train and y_test are labels
        Output: (N_train, 1) > importance of each training sample w.r.t. prediction of the model on the test set (X_test, y_test)
        '''
        torch.cuda.empty_cache()
        explainer = EstimatorNetwork().to(self.device) # X_train.shape[1] is the input feature dimensionality
        estimator = ParameterizedShapleyEstimator(alpha=self.alpha, beta=self.beta, explainer=explainer
                    , dataset_name = "fmnist", normalization=None
                    , model_checkpoint_dir = self.model_checkpoint_dir)
        
        # y_train_embeds = estimator.map_label_2_embeds(y_train)
        # y_test_embeds = estimator.map_label_2_embeds(y_test)
        
        train, valid = get_split_dataset(X_train, y_train)

        train = (train[0],train[1])    #(torch.Tensor(train[0]),torch.Tensor(train[1]))
        valid = (valid[0],valid[1])    #(torch.Tensor(valid[0]),torch.Tensor(valid[1]))
        test  = (X_test,y_test)        #(torch.Tensor(X_test), torch.Tensor(y_test))

        # train weighted shapley estimator
        if not os.path.exists(self.model_path):
            # estimator.train(train,
            #                 valid,
            #                 test,
            #                 lookback=20,
            #                 K=self.K,
            #                 verbose=True,
            #                 max_epochs=200,
            #                 eff_lambda=1e-2,
            #                 test_batch_size = 1000)

            print('===========================================')
            print('Number of Training samples: ', X_train.shape[0])
            print('Number of Test samples: ',X_test.shape[0])
            print('===========================================')
            # uncomment for using train data for evaluating utility
            estimator.train(train,
                            valid,
                            test,
                            lookback=20,
                            K=self.K,
                            verbose=True,
                            max_epochs=200,
                            eff_lambda=1e-2,
                            test_batch_size = 1000,
                            lr=1e-3)

            self.utility = estimator.utility_val
        else:
            print(f"Loading existing explainer model weights")
            explainer.load_state_dict(torch.load(self.model_path))
            explainer.eval()
        
        # if hasattr(self, "y_flipped") and self.y_flipped is not None:
        #     # y_flipped_embeds = estimator.map_label_2_embeds(self.y_flipped)
        #     mod_y = self.y_flipped
        # else:
        #     mod_y = y_train

        return estimator.inference(torch.Tensor(X_train)
                        ,torch.LongTensor(y_train)
                        ,torch.Tensor(X_test)
                        ,torch.LongTensor(y_test)
                        ,test_inf_batch_size
                        ).detach().cpu().numpy()

    def score(self, X_train, y_train, X_test, y_test, model_family='', model=None):
        '''
        This function gets called by shapley/utils/DShap.py, which is in turn called by different application scripts like shapley/apps/label.py
        '''
        if model is None:
            return self._get_shapley_value_torch(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

        self.model = model
        # print('Trained Model Accuracy: ',model.score(X_test, y_test))
        if model_family =="resnet18":
            model.eval()
            model.to(self.device)
            
            # uncomment for using the frozen feature extractor of the base resnet model
            # print("Extracting features from base model")
            # with torch.no_grad():
            #     X_feature =  self.extract_resnet_features(model, X_train)
            #     X_test_feature = self.extract_resnet_features(model, X_test) 

            # using resnet module inside the estimator network
            X_feature = X_train
            X_test_feature = X_test

        elif model_family == "NN":
            nn = model
            X_feature = np.maximum(np.matmul(X_train, nn.coefs_[0]) + nn.intercepts_[0], 0) 
            X_test_feature = np.maximum(np.matmul(X_test, nn.coefs_[0]) + nn.intercepts_[0], 0)

        return self._get_shapley_value_torch(X_train=X_feature, y_train=y_train, X_test=X_test_feature, y_test=y_test)