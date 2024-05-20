from shapley.apps import App
import numpy as np
import torchvision.models as models
import torch.nn as nn
from shapley.utils.shap_utils import return_model
from shapley.utils import DShap
import torch
import copy
import torch.optim as optim
from torch.optim import lr_scheduler
from shapley.utils.utils import batch
import time 
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
from shapley.utils.shap_utils import *
from sklearn.linear_model import LogisticRegression

class Summarization(App):

    def __init__(self, X, y, X_test, y_test, model_family='resnet18', model_checkpoint_dir="./checkpoints", **kwargs):
        self.name = 'Summarization'
        self.X = X
        self.y = y
        self.X_test = X_test
        self.y_test = y_test
        model_name = model_family

        self.num_train = len(X)
        self.num_test = len(X_test)
        self.model_name = model_name
        self.num_classes = np.max(self.y) + 1

        self.model = getattr(models, model_name)(pretrained=True)
        self.model.avgpool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_classes)
        self.model_checkpoint_dir = model_checkpoint_dir
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.feature_extractor_model = return_model(self.model_family, **kwargs).to(self.device)

        feature_extractor_model_path = f"{model_checkpoint_dir}/{model_family}_model_final.pth"
        if model_checkpoint_dir is not None and os.path.exists(feature_extractor_model_path):
            print(f"Loading existing {model_family} model weights")
            self.feature_extractor_model.load_state_dict(torch.load(feature_extractor_model_path))
            self.feature_extractor_model.eval()

    def run(self, measure,  x_ratio=0.2, batch_size=128, epochs=15, HtoL=False):
        

        dshap = DShap(X=self.X,
              y=self.y,
              X_test=self.X_test,
              y_test=self.y_test,
              num_test=self.num_test,
              model_family=self.model_name,
              measure=measure,
              model_checkpoint_dir = self.model_checkpoint_dir)
        

        # sx_train = torch.from_numpy(sx_train).contiguous().view(-1, 3,64,64)
        # sy_train = torch.from_numpy(sy_train).view(-1,).long()
        # sx_test = torch.from_numpy(sx_test).contiguous().view(-1, 3,64,64)
        # sy_test = torch.from_numpy(sy_test).view(-1,).long()

        # knn_value = measure.score(self.X, self.y, self.X_test, self.y_test, model=return_model(self.model_name).cuda())
        print("Getting data importance for all samples")
        start = time.time()
        knn_value = dshap.run(save_every=10, err = 0.5)
        print(f"Time taken: {time.time() - start}")
        
        print("Getting prediction accuracy when features are removed eith hihget to lowes tor lowest ot highest")
        count = int(len(self.X))
        interval = int(count * x_ratio)
        knn_accs = []
        idxs = np.argsort(knn_value.squeeze())
        keep_idxs = idxs.tolist()

        X_resnet_features = self.extract_resnet_features(self.X)
        for j in range(0, count, interval):
            if len(keep_idxs) == len(self.X):
                x_train_keep, y_train_keep = X_resnet_features, self.y
            else:
                x_train_keep, y_train_keep = X_resnet_features[keep_idxs], self.y[keep_idxs]

            knn_resnet = copy.deepcopy(self.model)
            knn_resnet = knn_resnet.to(self.device)
            optimizer = optim.SGD(knn_resnet.parameters(), lr=0.001, momentum=0.9)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
            criterion = nn.CrossEntropyLoss()

            acc = self.plot_tiny_train(x_train_keep, y_train_keep, self.X_test, self.y_test)
            acc = acc.cpu().detach().numpy()
            knn_accs.append(acc * 100.0)
            print(len(keep_idxs), acc)
            if(HtoL == True):
                keep_idxs = keep_idxs[:-interval] # removing data from highest to lowest
            else:
                keep_idxs = keep_idxs[interval:] # removing data from lowest to highest

        print(self.name, " acc:", len(keep_idxs), knn_accs)
        return knn_accs

    def extract_resnet_features(self, data_x):
        # data_x_tensor=  torch.from_numpy(data_x.reshape(data_x.shape[0],1,28,28)).repeat(1,3,1,1)

        pca = PCA(n_components=32)
        
        device = next(self.feature_extractor_model.parameters()).device
        data_loader = DataLoader(TensorDataset(data_x), 32, False)
        result = []
        for i, (x,) in enumerate(data_loader):
            x_features, _ =  self.feature_extractor_model(x.to(device))
            result.append(x_features)

        X_features = torch.cat(result, dim = 0).to("cpu")
        X_pca_features = pca.fit_transform(X_features)
        return X_pca_features

    def plot_tiny_train(self, x_train, y_train, x_test, y_test):
        model = LogisticRegression().fit(x_train, y_train)
        t_score = model.score(x_train, y_train)

        v_score = model.score(x_test, y_test)

        print('Train Acc: {:.4f}'.format(t_score))
        print('Valid Acc: {:.4f}'.format(v_score))
        return v_score

# def plot_tiny_train(phase, model, device, x_train, y_train, optimizer, criterion, scheduler, batch_size, epochs=1):
#     dataset_sizes = x_train.shape[0]
#     for epoch in range(epochs):
#         if phase == 'train':
#             scheduler.step()
#             model.train()  # Set model to training mode
#         else:
#             model.eval()   # Set model to evaluate mode

#         running_loss = 0.0
#         running_corrects = 0
#         # Iterate over data.
#         for inputs, labels in batch(x_train, y_train, batch_size):
#             inputs = torch.from_numpy(inputs.reshape(inputs.shape[0],1,28,28)).repeat(1,3,1,1).to(device)
#             labels = torch.from_numpy(labels).type(torch.LongTensor).to(device)
#             optimizer.zero_grad()
#             with torch.set_grad_enabled(phase == 'train'):
#                 outputs = model(inputs)
#                 _, preds = torch.max(outputs, 1)
#                 loss = criterion(outputs, labels)
#         # backward + optimize only if in training phase
#                 if phase == 'train':
#                     loss.backward()
#                     optimizer.step()
#             running_loss += loss.item() * inputs.size(0)
#             running_corrects += torch.sum(preds == labels.data)
#         epoch_loss = running_loss / dataset_sizes
#         epoch_acc = running_corrects.double() / dataset_sizes

#         print("\r{} epochs: {}/{}, Loss: {}, Acc: {}.".format(phase, epoch+1, epochs, epoch_loss, epoch_acc))
#         if phase == 'train':
#             # print("\r{} epochs: {}/{}, Loss: {}, Acc: {}.".format(phase, epoch+1, epochs, epoch_loss, epoch_acc), end="")
#             avg_loss = epoch_loss
#             t_acc = epoch_acc
#         elif phase == 'val':
#             val_loss = epoch_loss
#             val_acc = epoch_acc

#     if phase == 'train':
#         print('Train Loss: {:.4f} Acc: {:.4f}'.format(avg_loss, t_acc))
#         return
#     elif phase == 'val':
#         print('Val Loss: {:.4f} Acc: {:.4f}'.format(val_loss, val_acc))
#         return val_acc, val_loss
