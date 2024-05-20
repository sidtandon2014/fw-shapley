from shapley.apps import App
from shapley.utils import DShap
import numpy as np
from init import set_seed
set_seed()

class Label(App):
    '''
    Class for Noisy Label Detection Task
    '''

    def __init__(self, X, y, X_test, y_test, model_family='NN', model_checkpoint_dir="./checkpoints"):
        self.name = 'Label'
        self.X = X #.reshape((X.shape[0], -1))
        self.y = np.squeeze(y)
        self.X_test = X_test #.reshape((X_test.shape[0], -1))
        self.y_test = np.squeeze(y_test)
        self.num_train = len(X)
        self.num_flip = self.num_train // 10
        self.num_test = len(X_test)
        self.flip = None
        self.model_family = model_family
        self.model_checkpoint_dir = model_checkpoint_dir

        if self.flip is None:
            num_classes = np.max(self.y) + 1
            self.y_flipped = np.copy(self.y)
            flip_indice = np.random.choice(self.num_train, self.num_flip, replace=False)
            self.y_flipped[flip_indice] = (self.y_flipped[flip_indice] + 1) % num_classes

            self.flip = np.zeros(self.num_train)
            self.flip[flip_indice] = 1
        self.y = self.y_flipped

    def run(self, measure):
        
        # Set this measure only for FastWeightedShapley 
        # so that during training we can refer actual y
        # and during inferencing we can use flipped y
        # set_op = getattr(measure, "set", None)
        # if callable(set_op):
        #     print("Setting y flipped")
        #     set_op(y_flipped = self.y_flipped)
        # else:
        #     print("Warning! This should be called only with KNN Shapley")
        

        dshap = DShap(X=self.X,
              y=self.y,
              X_test=self.X_test,
              y_test=self.y_test,
              num_test=self.num_test,
              model_family=self.model_family,
              measure=measure,
              model_checkpoint_dir = self.model_checkpoint_dir)
        result = dshap.run(save_every=10, err = 0.5)
        # print('done!')
        # print('result shown below:')
        # print(result)
        return result
