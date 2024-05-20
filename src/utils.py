import math

import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from decimal import Decimal


class Utility():
    def __init__(self,metric='accuracy',num_classes=None) -> None:
        self.metric = metric
        self.map_metric_to_function = {'accuracy':self.get_accuracy,
                                       'entropy':self.get_entropy,
                                       'NLL':self.get_NLL}
        self.num_classes = num_classes
        
    def get_utility_KNN(self, X_train, y_train, X_test, y_test, K):
        '''
        Input: X_train - input features from training dataset [shape: N x d] (could be output of feature extractors)
                        N refers to number of samples from which we have to pick the nearest neighbors; d - #feats
            y_train - labels/output for samples in X_train [shape N x 1]
            X_test - input features of validation batch [shape: Nval x d]
            y_train - labels/output for samples in X_test [shape Nval x 1]

        Compute Time: 
            for N = 60k and Nval = 1k and d = 1024 (cifar10) - 14.1 secs
            for N = 60k and Nval = 100 and d = 1024 (cifar10) - 2 secs
        '''
        N = len(X_train)
        M = len(X_test)

        # step 1: for each validation pt x_test[i] find the nearest neighbors in X_train and compute p(y_j|x_test[i])
        p_hat = torch.zeros(M,self.num_classes)
        dist = torch.cdist(X_train.view(len(X_train), -1), X_test.view(len(X_test), -1))
        _, indices = torch.sort(dist, axis=0) # sort the values in each of the columns independently 
                                              # thus for every val pt we get the ranking of the training
                                              # pts w.r.t. euclidean distance
        del dist
        y_sorted = y_train[indices].reshape(-1, M) # labels of train pts w.r.t. sorted order
        y_sorted = y_sorted[:K] # we are interested in the top-K matched training pts

        # for each class i compute the class prob p(y_j|x_val) (parallely done for all val pts)
        # Sid: Added torch.sum
        for i in range(self.num_classes):
            p_hat[:,i] =  1/K * torch.sum((y_sorted == i).float(), dim = 0)

        # step 2: pass the true label/output and computed distribution over classes p(y_j|x_test[i]) (\forall j)
        #         to compute utility
        return self.map_metric_to_function[self.metric](p_hat,y_test)

    def get_accuracy(self,p_hat,y):
        '''
        (N = num of validation pts, C = num of classes)
        Input: p_hat - (N,C) i-th row contains the prob distribution over C classes for the i-th val pt
               y -(N,1) i-th row contains the true categorical label for the i-th val pt
        '''
        N = p_hat.shape[0]
        y = y.reshape(-1,1)
        assert y.shape==(N,1)

        # Sid: Modified below line
        # num_correct_preds = (torch.argmax(p_hat,axis=1).view(self.num_classes,1)==y).sum()
        num_correct_preds = (torch.argmax(p_hat,axis=1).view(N,1).to(y.device)==y).sum()
        return num_correct_preds/p_hat.shape[0]

    def get_entropy(self,p_hat,y):
        '''
        (N = num of validation pts, C = num of classes)
        Input: p_hat - (N,C) i-th row contains the prob distribution over C classes for the i-th val pt
               y -(N,1) i-th row contains the true categorical label for the i-th val pt
        '''
        entropy_per_val_pt = (-p_hat*torch.log(p_hat)).sum(axis=1)
        return entropy_per_val_pt.sum()

    def get_NLL(self,p_hat,y):
        '''
        (N = num of validation pts, C = num of classes)
        Input: p_hat - (N,C) i-th row contains the prob distribution over C classes for the i-th val pt
               y -(N,1) i-th row contains the true categorical label for the i-th val pt

        Output: for each val pt NLL = -log(p) of the true class 
                (https://ljvmiranda921.github.io/notebook/2017/08/13/softmax-and-the-negative-log-likelihood/)
        '''
        N = p_hat.shape[0]
        assert y.shape==(N,1)
        
        true_class_probs = p_hat[torch.arange(N), y.squeeze()]
        return (-torch.log(true_class_probs)).sum()


def w(M,a,b,j):
    '''
    Building the sampling distribution for weighted data shapley
    '''
    def beta_constant(a, b): # using Decimal module to deal with underflow of values
        '''
        the second argument (b; beta) should be integer in this function
        '''
        beta_fct_value=Decimal(1/a)
        for i in range(1,b):
                beta_fct_value=beta_fct_value*Decimal((i/(a+i)))
        return beta_fct_value
    
    return (M*beta_constant(j+b-1,M-j+a)/beta_constant(a,b))


def w_tilde(M,a,b,j): # use exponent based representation for handling 
    '''
    - w_tilde(j,a,b) = bin(M-1,j-1) * w(j,a,b) = T1 * T2
      We can do this approximation: express both T1 and T2 in the form: <a.bcdef x 10^{g}>
      Now, T1*T2 = (a1.b1c1d1e1f1 * a2.b2c2d2e2f2) * 10^{g1+g2}
    - We use the Decimal module to do this
    '''
    # exp_bc, base_bc = get_in_exp_form(binom_coeff(M-1,j-1))
    # exp_w, base_w = get_in_exp_form(w(M,a,b,j))
    #return ((base_bc*base_w)*math.pow(10,exp_bc+exp_w))
    return Decimal(math.comb(M-1,j-1))*Decimal(w(M,a,b,j))


def beta_shapley_subset_cardinality_wt(N,a,b,j): # for the subsets with cardinality j, this function gives the weight
    return Decimal(w_tilde(N,a,b,j))/Decimal((j*(N-j)))


def compute_weighted_shapley_wts(m, alpha=16, beta=1):
    weight_list=np.zeros(m)
    for j in range(1,m):
        weight_list[j-1] = beta_shapley_subset_cardinality_wt(m,alpha,beta,j)
    return weight_list/ np.sum(weight_list)


# def compute_weight_list(m, alpha=16, beta=1, compute_w_tilda = False):
#     '''
#     Given a prior distribution (beta distribution (alpha,beta))
#     beta_constant(j+1, m-j) = j! (m-j-1)! / (m-1)! / m # which is exactly the Shapley weights.

#     # weight_list[n] is a weight when baseline model uses 'n' samples (w^{(n)}(j)*binom{n-1}{j} in the paper).
#     '''

#     def beta_constant(a, b):
#         '''
#         the second argument (b; beta) should be integer in this function
#         '''
#         beta_fct_value=1/a
#         for i in range(1,b):
#             beta_fct_value=beta_fct_value*(i/(a+i))
#         return beta_fct_value


#     #import math
#     weight_list=np.zeros(m)
#     normalizing_constant=1/beta_constant(alpha, beta)
#     for j in np.arange(m):
#         # when the cardinality of random sets is j
#         den = beta_constant(j+1, m-j)
#         if den == 0:
#             return weight_list[:j]/ np.sum(weight_list[:j])

#         weight_list[j]=beta_constant(j+beta, m-j+alpha-1)/den
#         weight_list[j]=normalizing_constant*weight_list[j] # we need this '/m' but omit for stability # normalizing
#         if compute_w_tilda and j>1:
#             weight_list[j] = math.comb(m-1,j-1) * weight_list[j]
#     return weight_list/ np.sum(weight_list)


def save_model(model, path, device):
    model.cpu()
    torch.save(model.state_dict(), path)
    model.to(device)

# class WeightedShapleySampler:
#     '''
#     For sampling player subsets from the Shapley distribution.

#     Args:
#       num_players: number of players.
#     '''

#     def __init__(self, total_cardinality, alpha, beta):
#         weights = compute_weight_list(total_cardinality, alpha, beta)
#         self.categorical = Categorical(probs=torch.tensor(weights))
#         self.total_cardinality = total_cardinality
            

#     def sample(self):
#         '''
#         Generate sample.

#         Args:
#           batch_size: number of samples.
#           paired_sampling: whether to use paired sampling.
#         '''
#         return self.categorical.sample()

if __name__ == "__main__":

    num_classes = 10
    utility = Utility(num_classes=num_classes)

    train_instance= 60000
    test_instance = 10000
    dim = 512
    X_train = torch.from_numpy(np.random.random((train_instance,dim)))
    y_train = torch.from_numpy(np.random.choice(list(range(num_classes)),(train_instance,1)))


    X_test = torch.from_numpy(np.random.random((test_instance,dim)))
    y_test = torch.from_numpy(np.random.choice(list(range(num_classes)),(test_instance,1)))

    result = utility.get_utility_KNN(X_train,y_train,X_test,y_test, 5)

    print(result)
