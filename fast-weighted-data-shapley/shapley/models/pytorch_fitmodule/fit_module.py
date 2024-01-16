import torch

from collections import OrderedDict
from functools import partial
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, Module
from torch.optim import SGD
import numpy
from datetime import datetime
import os
from .utils import add_metrics_to_log, get_loader, log_to_message, ProgressBar
from copy import deepcopy

DEFAULT_LOSS = CrossEntropyLoss()
DEFAULT_OPTIMIZER = partial(SGD, lr=0.001, momentum=0.9)


class FitModule(Module):

    def eval_hessian(self, loss_grad):
        cnt = 0
        for g in loss_grad:
            g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector, g.contiguous().view(-1)])
            cnt = 1
        l = g_vector.size(0)
        hessian = torch.zeros(l, l)
        for idx in range(l):
            g_vector[idx].requires_grad = True
            grad2rd = torch.autograd.grad(g_vector[idx], self.parameters(), create_graph=True)
            cnt = 0
            for g in grad2rd:
                g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
                cnt = 1
            hessian[idx] = g2
        return hessian.cpu().data.numpy()

    def fit(self,
            X,
            y,
            batch_size=64,
            epochs=10,
            verbose=1,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            initial_epoch=0,
            seed=None,
            loss_fn=DEFAULT_LOSS,
            optimizer=DEFAULT_OPTIMIZER,
            metrics=None,
            model_family = "",
            model_checkpoint_dir = ""):
        """Trains the model similar to Keras' .fit(...) method

        # Arguments
            X: training data Tensor.
            y: target data Tensor.i
            batch_size: integer. Number of samples per gradient update.
            epochs: integer, the number of times to iterate
                over the training data arrays.
            verbose: 0, 1. Verbosity mode.
                0 = silent, 1 = verbose.
            validation_split: float between 0 and 1:
                fraction of the training data to be used as validation data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate
                the loss and any model metrics
                on this data at the end of each epoch.
            validation_data: (x_val, y_val) tuple on which to evaluate
                the loss and any model metrics
                at the end of each epoch. The model will not
                be trained on this data.
            shuffle: boolean, whether to shuffle the training data
                before each epoch.
            initial_epoch: epoch at which to start training
                (useful for resuming a previous training run)
            seed: random seed.
            optimizer: training optimizer
            loss: training loss
            metrics: list of functions with signatures `metric(y_true, y_pred)`
                where y_true and y_pred are both Tensors

        # Returns
            list of OrderedDicts with training metrics
        """
        if seed and seed >= 0:
            torch.manual_seed(seed)
        # Prepare validation data
        if validation_data:
            X_val, y_val = validation_data
        elif validation_split and 0. < validation_split < 1.:
            split = int(X.size()[0] * (1. - validation_split))
            X, X_val = X[:split], X[split:]
            y, y_val = y[:split], y[split:]
        else:
            X_val, y_val = None, None
        # Build DataLoaders
        if isinstance(X, numpy.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(y, numpy.ndarray):
            y = torch.from_numpy(y).long()
        if isinstance(X_val, numpy.ndarray):
            X_val = torch.from_numpy(X_val).float()
        if isinstance(y_val, numpy.ndarray):
            y_val = torch.from_numpy(y_val).long()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        num_train = X.shape[0]
        num_valid = X_val.shape[0]

        train_loader = get_loader(X, y, batch_size, shuffle)
        val_loader = get_loader(X_val, y_val, batch_size, shuffle)

        # Compile optimizer
        opt = optimizer(self.parameters())
        best_vloss = float('inf')
        # Run training loop
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        print(f"Model device {next(self.parameters()).device}")
        
        for epoch in range(epochs):
            if verbose:
                print("Epoch {0} / {1}".format(epoch+1, epochs))
            
            running_loss = 0.0
            train_acc = 0.0
            # Run batches
            self.train()
            for batch_i, batch_data in enumerate(train_loader):
                # Get batch data
                X_batch = batch_data[0].to(device)
                y_batch = batch_data[1].to(device)
                # Backprop
                opt.zero_grad()
                (_, y_batch_pred) = self(X_batch)

                batch_loss = loss_fn(y_batch_pred, y_batch)
                batch_loss.backward()
                opt.step()
                # Update status
                loss = batch_loss.item()
                
                running_loss += loss
                _, predicted = torch.max(y_batch_pred, 1)
                train_acc += (predicted == y_batch).float().sum().item()

            train_acc = 100 * train_acc/ num_train
            avg_loss = running_loss / (batch_i + 1)

            running_vloss = 0.0
            valid_acc = 0
            self.eval()
            with torch.no_grad():
                avg_vloss, valid_acc = self.validate(val_loader, loss_fn, self, num_valid, device)
                # for i, (vdata_x, vdata_y) in enumerate(val_loader):
                #     vdata_x, vdata_y  = vdata_x.to(device), vdata_y.to(device)

                #     (_, v_predict) = self(vdata_x)
                #     vloss = loss_fn(v_predict, vdata_y)
                    
                #     running_vloss += vloss.item()
                #     _, predicted = torch.max(v_predict, 1)
                #     valid_acc += (vdata_y == predicted).float().sum().item()
                
                # valid_acc = 100 * valid_acc/ num_valid

            # avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} loss valid {}'.format(avg_loss, avg_vloss))
            print(f"Train acc {train_acc} Valid ACC {valid_acc}")
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = '{}_model_{}_{}'.format(model_family,timestamp, epoch)
                if not os.path.exists(os.path.join(model_checkpoint_dir,"intermediate")):
                    os.makedirs(os.path.join(model_checkpoint_dir,"intermediate"))
                torch.save(self.state_dict(), os.path.join(model_checkpoint_dir,"intermediate", model_path))
                best_model = deepcopy(self)
        
        _, valid_acc = self.validate(val_loader, loss_fn, best_model, num_valid, device)
        print(f"Best model Valid ACC {valid_acc}")
        torch.save(best_model.state_dict(), os.path.join(model_checkpoint_dir,f"{model_family}_model_final.pth"))

    def validate(self, val_loader, loss_fn, model, num_valid, device):
        valid_acc = 0
        running_vloss = 0.0
        for i, (vdata_x, vdata_y) in enumerate(val_loader):
            vdata_x, vdata_y  = vdata_x.to(device), vdata_y.type(torch.LongTensor).to(device)

            (_, v_predict) = model(vdata_x)
            vloss = loss_fn(v_predict, vdata_y)
            
            running_vloss += vloss.item()
            _, predicted = torch.max(v_predict, 1)
            valid_acc += (vdata_y == predicted).float().sum().item()

        valid_acc = 100 * valid_acc/ num_valid
        avg_vloss = running_vloss / (i + 1)

        return avg_vloss, valid_acc

    def evaluate(self, X_test, y_test):
        val_loader = get_loader(X_test, y_test, 32, shuffle=False)
        device = next(self.parameters()).device
        with torch.no_grad():
            _, acc = self.validate(val_loader,DEFAULT_LOSS,self,X_test.shape[0], device)
        return acc
            
    def predict(self, X, batch_size=32):
        """Generates output predictions for the input samples.

        Computation is done in batches.

        # Arguments
            X: input data Tensor.
            batch_size: integer.

        # Returns
            prediction Tensor.
        """
        # Build DataLoader
        data = get_loader(X, batch_size=batch_size)
        # Batch prediction
        self.eval()
        r, n = 0, X.size()[0]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for batch_data in data:
            # Predict on batch
            X_batch = batch_data[0].to(device)
            y_batch_pred = self(X_batch)
            # Infer prediction shape
            if r == 0:
                y_pred = (torch.zeros((n,) + y_batch_pred.size()[1:])).data.type('torch.FloatTensor')
            # Add to prediction tensor
            y_pred[r : min(n, r + batch_size)] = y_batch_pred
            r += batch_size
        return y_pred
