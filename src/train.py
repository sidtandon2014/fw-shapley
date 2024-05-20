from copy import deepcopy
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from datasets import (get_tensor_dataset,
                      get_weighted_data_shapley_dataset, get_sampler, StratifiedBatchSampler, CustomTensorDataset)
from utils import Utility, save_model
from init import *
from transformers import CLIPTokenizer, CLIPTextModelWithProjection
from init import set_seed
set_seed()

label_embeds = {}
def additive_efficient_normalization(pred, grand, null):
    '''
    Apply additive efficient normalization.

    Args:
      pred: model predictions.
      grand: grand coalition value.
      null: null coalition value.
    '''
    gap = (grand - null) - torch.sum(pred)
    # gap = gap.detach()
    return pred + gap.unsqueeze(1) / pred.shape[0]


def multiplicative_efficient_normalization(pred, grand, null):
    '''
    Apply multiplicative efficient normalization.

    Args:
      pred: model predictions.
      grand: grand coalition value.
      null: null coalition value.
    '''
    ratio = (grand - null) / torch.sum(pred, dim=1)
    # ratio = ratio.detach()
    return pred * ratio.unsqueeze(1)


def calculate_grand_coalition(data, test_data, utility: Utility, K):
    '''
    Calculate the value of grand coalition for each input.
    i.e. Performance of Utility function on validation set when entire 
    training set is provided

    Args:
      dataset: dataset object.
      imputer: imputer model.
      batch_size: minibatch size.
      num_players: number of players.
      link: link function.
      device: torch device.
      num_workers: number of worker threads.
    '''
    with torch.no_grad():
        utility_perf = []
        batch_size = data[0].size(0)
        for index in range(batch_size):
            perf = utility.get_utility_KNN(
                data[0][index], data[1][index], test_data[0].squeeze(0), test_data[1].squeeze(0), K)
            utility_perf.append(perf)
        try:
            utility_perf = torch.stack(utility_perf, dim=0).reshape(-1, 1)
        except:
            print("error")
    return utility_perf


def calculate_null_coalition(total_labels):
    with torch.no_grad():
        null_val = 1.0 / total_labels.shape[0]
    return null_val

def get_label_embeddings():
    version = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(version)
    text_model = CLIPTextModelWithProjection.from_pretrained(version)
    
    with torch.no_grad():
        for key,val in id2label.items():
            batch_encoding = tokenizer(val, truncation=True, max_length=5, return_length=True,
                                                return_overflowing_tokens=True, padding="max_length", return_tensors="pt")
            tokens = batch_encoding["input_ids"]
            text_model_output = text_model(input_ids=tokens
                                        , attention_mask = batch_encoding["attention_mask"]
                                        , output_hidden_states=False)
            label_embeds[key] = text_model_output.text_embeds

def map_label_2_embeds(y, device):
    y_embeds = []
    for i in range(len(y)):
        y_embeds.append(label_embeds[y[i].item()])
    
    return torch.concat(y_embeds, dim = 0).to(device)

def evaluate_explainer(explainer, normalization, X_train, y_train, X_test, y_test, grand, null):
    '''
    Helper function for evaluating the explainer model and performing necessary
    normalization and reshaping operations.

    Args:
      explainer: explainer model.
      normalization: normalization function.
      x: input.
      grand: grand coalition value.
      null: null coalition value.
      num_players: number of players.
      inference: whether this is inference time (or training).
    '''
    # Evaluate explainer.
    # EFF_BATCH_SIZE will be applicable during inference
    # Why? Because during train and valid, X.size(0) will always be a multiple of SAMPLE_BATCH_SIZE
    # But during inference there is no such requirement and we can go with vanilla data loader with batch_size
    # as SAMPLE_BATCH_SIZE but the last batch might be less than SAMPLE_BATCH_SIZE
    # device = next(explainer.parameters()).device
    if X_train.size(0) % SAMPLE_BATCH_SIZE == 0:
        EFF_BATCH_SIZE = SAMPLE_BATCH_SIZE
    else:
        EFF_BATCH_SIZE = X_train.size(0)

    # EFF_BATCH_SIZE = SAMPLE_BATCH_SIZE if X_train.size(
    #     0) >= SAMPLE_BATCH_SIZE else X_train.size(0)

    # y_mod_train = map_label_2_embeds(y_train, device)
    # y_mod_test = map_label_2_embeds(y_test, device)

    sup_con_loss, pred = explainer(X_train, y_train, X_test, y_test)
    pred_batched = pred.reshape(EFF_BATCH_SIZE, X_train.size(
        0) // EFF_BATCH_SIZE, pred.size(-1))
    total = torch.sum(pred_batched, dim=1)
    
    # Apply normalization.
    if normalization:
        pred = normalization(pred, grand, null)

    return pred, total, sup_con_loss

def validate_min_count_per_class(y):
    counts = np.bincount(y)
    assert np.all(counts >= 2), "Validation min count per class failed"

class ParameterizedShapleyEstimator():
    def __init__(self, alpha, beta, explainer, dataset_name, normalization=None, model_checkpoint_dir="./checkpoints") -> None:
        # self. weight_list = compute_weight_list(n, alpha, beta)
        self.alpha, self.beta = alpha, beta
        self.utility_val = []
        self.train_batch_size = []
        self.model_checkpoint_dir = model_checkpoint_dir
        if not os.path.exists(self.model_checkpoint_dir):
            print(self.model_checkpoint_dir)
            os.makedirs(self.model_checkpoint_dir)
        self.explainer = explainer

        if dataset_name == "fmnist":
            self.num_classes = 10
        # Set up normalization.
        if normalization is None or normalization == 'none':
            self.normalization = None
        elif normalization == 'additive':
            self.normalization = additive_efficient_normalization
        elif normalization == 'multiplicative':
            self.normalization = multiplicative_efficient_normalization
        else:
            raise ValueError('unsupported normalization: {}'.format(
                normalization))

        # get_label_embeddings()

        
    def insert_class_info_into_x(self, X_train, y_train):

        assert len(X_train.shape) == 2, "Expected 2 dimensional training dataset"
        batch = X_train.shape[0]

        batch_size = X_train.size(0)
        W = H = int(X_train.shape[1] ** 0.5)
        ones = torch.ones((W, H))
        zeros = torch.zeros((batch_size, 10, W, H))
        X_train_with_classes = X_train.reshape(
            batch_size, W, H).unsqueeze(1)

        X_train_with_classes = torch.concat(
            [X_train_with_classes, zeros], dim=1)
        for index, y in enumerate(y_train):
            X_train_with_classes[index, y.item()+1, :, :] = ones

        return X_train_with_classes

    def set_classidxs(self, train_ds, test_ds, num_classes):
        # classidxs = np.random.choice(num_classes,NUM_CLASSES_PER_BATCH, False)
        classidxs = np.random.choice(num_classes,np.random.choice(np.arange(2,10)), False)
        train_ds.classidxs = classidxs
        test_ds.classidxs = classidxs
        
    def validate(self, val_loader, val_test_loader, utility, explainer, grand_val, null_value, normalization, loss_fn, K=5):
        '''
        Calculate mean validation loss.

        Args:
        val_loader: validation data loader.
        test_data: imputer model.
        explainer: explainer model.
        null: null coalition value.
        link: link function.
        normalization: normalization function.
        '''

        with torch.no_grad():
            # Setup.
            device = next(explainer.parameters()).device
            valid_loss = 0.
            grand_val = grand_val.to(device)
            null_value = null_value.to(device)

            
            with tqdm(val_loader, unit="batch") as vepoch:
                vepoch.set_description("Validation")
                for val_x, val_y in vepoch:
                    batch_loss = 0.
                    val_x_batched = val_x.unsqueeze(0)
                    val_y_batched = val_y.unsqueeze(0)
                    # NUM_PLAYERS = val_x.size(0) // SAMPLE_BATCH_SIZE
                    
                    # val_x_batched = val_x.reshape(
                    #     SAMPLE_BATCH_SIZE, NUM_PLAYERS, val_x.size(-1))
                    # val_y_batched = val_y.unsqueeze(-1).reshape(
                    #     SAMPLE_BATCH_SIZE, NUM_PLAYERS, 1)
                    
                    val_x = val_x.to(device)
                    val_y = val_y.to(device)
                    for test_x, test_y in val_test_loader:
                        # validate_min_count_per_class(test_y)
                        grand_subset = calculate_grand_coalition(
                            (val_x_batched, val_y_batched), (test_x, test_y), utility, K).to(device)

                        # val_x_with_class = self.insert_class_info_into_x(val_x, val_y)
                        # val_x_with_class = val_x_with_class.to(device)

                        # test_x = self.insert_class_info_into_x(test_x, test_y)
                        test_x = test_x.to(device)
                        test_y = test_y.to(device)
                        
                        # Evaluate explainer.
                        _, total, sup_con_loss = evaluate_explainer(
                            explainer, normalization, val_x, val_y, test_x, test_y, grand_val, null_value)
                        approx = total.sum(dim = 0).view(-1,1)
                        try:
                            loss = 1.0 * (sup_con_loss + loss_fn(grand_subset, approx).item())
                            # if torch.isnan(loss):
                            #     print("Nan occured")
                        except:
                            raise ValueError("Error in loss function")

                        batch_loss += loss
                    
                    valid_loss += batch_loss / len(val_test_loader)

                valid_loss = valid_loss/ len(val_loader)
                vepoch.set_postfix({"epoch_loss": valid_loss})
                vepoch.close()

        return valid_loss

    def inference(self, X_train, y_train, X_test, y_test, test_inf_batch_size):
        with torch.no_grad():
            device = next(self.explainer.parameters()).device
            train_loader = DataLoader(get_tensor_dataset(
                (X_train, y_train)), batch_size=INFERENCE_SAMPLE_BATCH_SIZE, shuffle=False)
            results = []
            test_loader = DataLoader(get_tensor_dataset(
                (X_test, y_test)), batch_size=test_inf_batch_size, shuffle=False)

            with tqdm(train_loader, unit="batch") as tepoch:
                tepoch.set_description("Inference")
                for i, (train_x, train_y) in enumerate(tepoch):
                    batch_results = None
                    # train_x = self.insert_class_info_into_x(train_x, train_y)
                    train_x = train_x.to(device)
                    train_y = train_y.to(device)
                    for test_x, test_y in test_loader:
                        # test_x = self.insert_class_info_into_x(test_x, test_y)
                        test_x = test_x.to(device)
                        test_y = test_y.to(device)
                        
                        pred, _, _ = evaluate_explainer(
                            self.explainer, None, train_x, train_y, test_x, test_y, grand=None, null=None)
                        if batch_results is not None:
                            batch_results = torch.concat([batch_results, pred], dim = -1)
                        else:
                            batch_results = pred #torch.mean(attn, dim = -1).view(-1,1)
                        # pred = pred.reshape(-1, 1)
                    agg_results = torch.concat([
                                    torch.mean(batch_results,dim = -1).view(-1,1)
                                    ,torch.sum(batch_results,dim = -1).view(-1,1)
                                    ,torch.max(batch_results,dim = -1)[0].view(-1,1)
                                    ], dim = -1)
                    
                    results.append(agg_results)
                tepoch.close()

        return torch.concat(results, 0)

    def train(self,
              train_data,
              val_data,
              test_data,
              max_epochs,
              K=5,
              lr=2e-4,
              min_lr=1e-5,
              lr_factor=0.5,
              eff_lambda=0,
              lookback=5,
              val_batch_size = 128,
              test_batch_size=1000,
              num_workers=0,
              verbose=False,
              ):
        '''
        Train explainer model.

        Args:
          train_data: training data with inputs only (np.ndarray, torch.Tensor).
          val_data: validation data with inputs only (np.ndarray, torch.Tensor).
          test_data: test data with inputs only (np.ndarray, torch.Tensor).
          batch_size: minibatch size.
          max_epochs: max number of training epochs.
          K: Hyper parameter for KNN utility
          lr: initial learning rate.
          min_lr: minimum learning rate for scheduler.
          lr_factor: learning rate decrease factor.
          eff_lambda: lambda hyperparameter for efficiency penalty.
          lookback: lookback window for early stopping.
          test_batch_size: Number of samples in testing batcj.
          num_workers: number of worker threads in data loader.
          verbose: verbosity.
        '''
        # Set up explainer model for training
        explainer = self.explainer  # estimator model
        normalization = self.normalization
        explainer.train()

        device = next(explainer.parameters()).device

        # null value computation: compute utility when the model is not trained on any data (random model's performance)
        if isinstance(test_data, tuple) and isinstance(test_data[0], np.ndarray):
            test_data = (torch.tensor(
                test_data[0]), torch.tensor(test_data[1]))
        if isinstance(train_data, tuple):
            unique_classes = torch.unique(torch.tensor(train_data[1]))
            self.num_classes = unique_classes.shape[0]
            # for acc utility -> null value performance on test set w.r.t. a random model => 1/C
            null_value = torch.tensor(
                calculate_null_coalition(unique_classes)).to(device)
        else:
            raise ValueError(
                "Unable to calcualte null value as train_data is not in right format")

        # init the utility class
        utility = Utility(num_classes=self.num_classes)

        # Set up validation dataset.
        print("Setting up validation datasets and dataloaders")
        assert val_batch_size > K * self.num_classes, "Validation batch size should be greater than (K * #classes), so that utility function has enough representation"
        val_set = get_tensor_dataset(val_data)
        val_loader = DataLoader(val_set, 
                                 pin_memory=True
                                 , num_workers=num_workers
                                 , batch_sampler=StratifiedBatchSampler(val_data[1]
                                                                        , batch_size=val_batch_size
                                                                        , shuffle=True))

        val_test_set = get_tensor_dataset((
            val_data[0].numpy().copy()
            ,val_data[1].numpy().copy()))
        val_test_loader = DataLoader(val_test_set, 
                                 pin_memory=True
                                 , num_workers=num_workers
                                 , batch_sampler=StratifiedBatchSampler(val_data[1]
                                                                        , batch_size=test_batch_size
                                                                        , shuffle=True))

        train_test_set = CustomTensorDataset(
            train_data[0].numpy().copy()
            ,train_data[1].numpy().copy()
            ,test_batch_size
        )
        train_test_loader = DataLoader(train_test_set
                                , pin_memory=True
                                , num_workers=num_workers)
        
        train_test_iter = iter(train_test_loader)

        # Grand coalition value.
        print("Calculating train grand coalition")
        # grand_train = calculate_grand_coalition(
        #     train_data, test_data, utility, K).to(device)
        grand_train = torch.tensor(0.9)

        print("Calculating valid grand coalition")
        # grand_val = calculate_grand_coalition(
        #     val_data, test_data, utility, K).to(device)
        grand_val = torch.tensor(0.9)

        # Setup for training.
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(explainer.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=lr_factor, patience=lookback // 2, min_lr=min_lr,
            verbose=verbose)
        self.loss_list = []
        best_loss = np.inf
        best_epoch = -1
        best_model = None

        sampler = get_sampler(train_data[0], self.alpha, self.beta)

        print("Start training")
        for epoch in range(max_epochs):
            # Batch iterable.
            # val_set = get_weighted_data_shapley_dataset(
            #     val_data, sampler, CONST_BATCH_SIZE=None)
            # val_loader = DataLoader(val_set, batch_size=1,
            #                         shuffle=True, num_workers=0)
            train_set = get_weighted_data_shapley_dataset(
                train_data, sampler, CONST_BATCH_SIZE=None)
            train_loader = DataLoader(train_set, batch_size=1,
                                      shuffle=True, num_workers=0)

            N = 0
            train_loss = 0.

            # Set class_idxs in dataset 
            self.set_classidxs(train_set, train_test_set, self.num_classes)
            with tqdm(train_loader, unit="batch") as tepoch:
                tepoch.set_description(f"Training epoch: {epoch}")
                for train_x, train_y in tepoch:
                    train_x = train_x.squeeze(0)  # [b, dim]
                    train_y = train_y.squeeze(0)  # [b]
                    NUM_PLAYERS = train_x.size(0) // SAMPLE_BATCH_SIZE

                    try:
                        train_test_batch = next(train_test_iter)
                    except StopIteration:
                        train_test_iter = iter(train_test_loader)
                        train_test_batch = next(train_test_iter)

                    # validate_min_count_per_class(train_test_batch[1])
                    # print('Test batch size:', test_batch[0].shape)

                    with torch.no_grad():
                        train_x_batched = train_x.reshape(SAMPLE_BATCH_SIZE, train_x.size(
                            0) // SAMPLE_BATCH_SIZE, train_x.size(-1))
                        train_y_batched = train_y.unsqueeze(-1).reshape(
                            SAMPLE_BATCH_SIZE, train_y.size(0) // SAMPLE_BATCH_SIZE, 1)
                        grand_subset = calculate_grand_coalition(
                            (train_x_batched, train_y_batched), train_test_batch, utility, K).to(device)
                        
                        # Debug: Analyze how much the utility function varies
                        self.utility_val.append(grand_subset)
                        self.train_batch_size.append(train_x_batched.shape[1])

                    # train_x = self.insert_class_info_into_x(train_x, train_y)
                    # Move to device.
                    train_x = train_x.to(device)
                    train_y = train_y.to(device)
                    # train_y = train_y.to(device)
                    test_x = train_test_batch[0].squeeze(0)
                    test_y = train_test_batch[1].squeeze(0)
                    # test_x = self.insert_class_info_into_x(test_x, test_y)
                    test_x = test_x.to(device)
                    test_y = test_y.to(device)

                    _, total, sup_con_loss = evaluate_explainer(
                        explainer, normalization, train_x, train_y, test_x, test_y, grand_subset, null_value)

                    # torch.sum(pred, dim = 0) # null_value + pred
                    approx = total
                    try:
                        loss = 1.0 * (sup_con_loss + loss_fn(grand_subset, approx))
                    except:
                        raise ValueError("Error in loss function")
                    # if eff_lambda:
                    #     # total = evaluate_explainer(explainer, normalization, train_x, grand_train, null_value)
                    #     loss = loss + eff_lambda * loss_fn(grand_train.repeat(approx.shape), approx)

                    loss.backward()
                    optimizer.step()
                    explainer.zero_grad()

                    # N += len(train_x)
                    train_loss += loss.item()

                    # Set class_idxs in dataset 
                    self.set_classidxs(train_set, train_test_set, self.num_classes)

                train_loss = train_loss / len(train_loader)
                tepoch.set_postfix({"epoch_loss": train_loss})
                tepoch.close()

            # Evaluate validation loss.
            explainer.eval()

            val_loss = self.validate(
                val_loader, val_test_loader, utility, explainer, grand_val, null_value, normalization, loss_fn)
            explainer.train()

            # Save loss, print progress.
            if verbose & (epoch % 1 == 0):
                print('----- Epoch = {} -----'.format(epoch + 1))
                print('Train loss = {:.6f} Val loss = {:.6f}'.format(
                    train_loss, val_loss))
                print('')
            scheduler.step(val_loss)
            self.loss_list.append(val_loss)

            # Check for convergence.
            if self.loss_list[-1] < best_loss:
                best_loss = self.loss_list[-1]
                best_epoch = epoch
                best_model = deepcopy(explainer)
                current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                if not os.path.exists(os.path.join(self.model_checkpoint_dir, "intermediate")):
                    os.makedirs(os.path.join(
                        self.model_checkpoint_dir, "intermediate"))

                model_chkpoint_path = f"{self.model_checkpoint_dir}/intermediate/explainer_{current_time}_{epoch}_{best_loss}.pth"
                save_model(best_model, model_chkpoint_path, device)
                if verbose:
                    print(f'New best epoch: {epoch+1}, loss = {val_loss}')
                    print('')
            elif epoch - best_epoch == lookback:
                if verbose:
                    print('Stopping early at epoch = {}'.format(epoch))
                break

        # Copy best model.
        for param, best_param in zip(explainer.parameters(),
                                     best_model.parameters()):
            param.data = best_param.data

        print("Saving best model")
        model_chkpoint_path = f"{self.model_checkpoint_dir}/explainer_model_final.pth"
        save_model(best_model, model_chkpoint_path, device)
        explainer.eval()
