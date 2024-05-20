from shapley.apps import App
from shapley.utils import DShap
import numpy as np
from sklearn.linear_model import LogisticRegression
import os
import torch

class InclusionExclusion(App):
    def __init__(self, X, y, X_test, y_test
                , method_name = "fw_shapley"
                , app_name = 'inclusion_exclusion'
                , dataset = "fmnist"
                , model_family='NN'
                , evaluation_model = "logistic"
                , model_checkpoint_dir="./checkpoints"):
        self.app_name = app_name

        self.results_dir = f"../../results/{self.app_name}/{dataset}/"
        if not os.path.exists(self.results_dir):
            os.mkdir(self.results_dir)

        self.X = X # X.reshape((X.shape[0], -1))
        self.y = np.squeeze(y)
        self.X_test = X_test # .reshape((X_test.shape[0], -1))
        self.y_test = np.squeeze(y_test)
        self.num_classes = np.max(self.y) + 1
        self.num_train = len(X)
        self.num_test = len(X_test)
        self.model_family = model_family
        self.model_checkpoint_dir = model_checkpoint_dir
        self.inclusion_exclusion_evaluation_model = self.get_model(evaluation_model)
        self.method_name = method_name

    def generate_sample_data(self):
        sample_data = {}
        for class_idx in np.arange(self.num_classes):
            data_indexes = np.where([self.y_embeddings == class_idx])
            np.random.shuffle(data_indexes) 
            sample_data[class_idx] = self.X_embeddings[data_indexes[0]].copy()
        return sample_data

    def run(self, measure):
        
        dshap = DShap(X=self.X,
              y=self.y,
              X_test=self.X_test,
              y_test=self.y_test,
              num_test=self.num_test,
              model_family=self.model_family,
              measure=measure,
              model_checkpoint_dir = self.model_checkpoint_dir)

        shap_scores_file = f"{self.results_dir}/{self.method_name}_shap_scores.npy"
        # if os.path.exists(shap_scores_file):
        #     print("Loading SHAP scores")
        #     np.load(shap_scores_file)
        # else:
        print("Calculating SHAP scores")
        shap_scores = dshap.run(save_every=10, err = 0.5)
        np.save(shap_scores_file, shap_scores)

        self.X_embeddings, self.y_embeddings  = dshap.embeddings
        self.X_test_embeddings, self.y_test_embeddings  = dshap.test_embeddings

        # Generate samples data
        # self.sample_data = self.generate_sample_data()
        # scores = {"mean":shap_scores[:,0],"sum":shap_scores[:,1], "max":shap_scores[:,2]}

        if self.method_name == "fw_shapley":
            print("Picking up sum of weighted shapley scores")
            scores = shap_scores[:,1]
        else:
            scores = shap_scores #[:,0]
        if isinstance(scores, torch.Tensor):
            scores = scores.numpy()
        point_exclusion_scores = self.point_exclusion_exp(scores)
        # point_inclusion_scores = self.point_inclusion_exp(scores)
        
        exclusion_results_file = f"{self.results_dir}/{self.method_name}_exc_scores.npy"
        # inclusion_results_file = f"{self.results_dir}/{self.method_name}_inc_scores.npy"

        print("Saving inclusion exclusion results")
        np.save(exclusion_results_file, point_exclusion_scores)
        # np.save(inclusion_results_file, point_inclusion_scores)

        return point_exclusion_scores


    def get_model(self, mode, **kwargs):
        if mode=='logistic':
            solver = kwargs.get('solver', 'liblinear')
            n_jobs = kwargs.get('n_jobs', -1)
            C = kwargs.get('C', 0.05) # 1.
            max_iter = kwargs.get('max_iter', 5000)
            model = LogisticRegression(solver=solver, n_jobs=n_jobs, C=C,
                                    max_iter=max_iter, random_state=666)

            return model
        
    def point_inclusion_exclusion_exp_core(self, shap_scores_index_desc, percentiles):
        # Data points are included/ excluded based on percentile until minimum samples remain
        
        X_init, y_init=self.X_embeddings, self.y_embeddings
        self.inclusion_exclusion_evaluation_model.fit(X_init, y_init) # performance with all data points
        vals=[(100, self.inclusion_exclusion_evaluation_model.score(X=self.X_test_embeddings, y=self.y_test_embeddings))]

        for top_k_sources in percentiles:

            total_points_to_consider = int(self.num_train * top_k_sources/ 100)
            
            X_subset = self.X_embeddings[shap_scores_index_desc[:(total_points_to_consider+1)].copy()]
            y_subset = self.y_embeddings[shap_scores_index_desc[:(total_points_to_consider+1)].copy()]

            
            self.inclusion_exclusion_evaluation_model.fit(X_subset, y_subset)
            vals.append((top_k_sources, self.inclusion_exclusion_evaluation_model.score(X=self.X_test_embeddings, y=self.y_test_embeddings)))

        return np.array(vals) 

    def point_exclusion_exp(self, shap_scores):
        # Exclude least important data point first (value order: decreasing)
        percentiles = [99, 95, 90, 85, 75, 50, 25, 15, 10, 5, 1]
        vals=self.point_inclusion_exclusion_exp_core(np.argsort(shap_scores)[::-1], percentiles)

        return vals

    def point_inclusion_exp(self, shap_scores):
        # Include high important data point first (value order: decreasing)
        percentiles = [5, 10, 15, 30, 50, 60, 70, 80, 90, 99]
        vals=self.point_inclusion_exclusion_exp_core(np.argsort(shap_scores)[::-1], percentiles)

        return vals