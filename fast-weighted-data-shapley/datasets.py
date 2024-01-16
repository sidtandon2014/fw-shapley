import gzip

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms
import math
import utils
from torch.distributions import Categorical
from init import *
from sklearn.model_selection import StratifiedKFold
from collections import Counter

def extract_features(feature_extractor, dataset):
    transform_fmnist = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # batch_size = 128
    # total_batches = math.ceil(len(data) * 1.0 / batch_size)
    extracted_data = []
    device = next(feature_extractor.parameters()).device
    
    for data in dataset:
        subset = transform_fmnist(data).to(device)
        with torch.no_grad():
            tmp = feature_extractor(subset.unsqueeze(0)).cpu()
        extracted_data.append(tmp) 

    final_ds = torch.stack(extracted_data, dim=0)
    return final_ds.squeeze(1)

def get_split_dataset(data, labels):
    split1 = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    for train_index, test_valid_index in split1.split(data, labels):
        train_set = data[train_index], labels[train_index]
        valid_set = data[test_valid_index], labels[test_valid_index]

    return train_set, valid_set

def load_datasets(dataset_type, paths):
    if dataset_type.lower() == "fashionmnist":
        return load_FashionMNIST(paths)
    else: 
        raise("Not a valid dataset")


def load_FashionMNIST(paths):
    data = {}
    for (root_path, kind) in paths:
        
        with gzip.open(f"{root_path}/{kind}-labels-idx1-ubyte.gz", 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                                    offset=8)

        with gzip.open(f"{root_path}/{kind}-images-idx3-ubyte.gz", 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                offset=16).reshape(len(labels), 28, 28)

        data[kind] = (images, labels)
    return data

def get_sampler(data, alpha, beta):
    indexes = list(range(data.shape[0]))
    # todo: Sid Modified this to consider only top indexes

    total_indexes = len(indexes)
    if total_indexes > MAX_DATA_SAMPLES:
        total_indexes = MAX_DATA_SAMPLES

    w = torch.Tensor(utils.compute_weighted_shapley_wts(m=total_indexes,alpha=alpha,beta=beta))
    w[MAX_DATA_SAMPLES_PER_BATCH:] = 0
    sampler = Categorical(probs = w)
    return sampler

def get_batch_indexes(data, sampler, CONST_BATCH_SIZE = None):
    '''
    Given the data it creates dynamic batch size using weighted beta sampling (alpha, beta)
    '''
    
    if isinstance(data, np.ndarray) or isinstance(data, torch.Tensor):
        indexes = list(range(data.shape[0]))
    elif isinstance(data, pd.DataFrame):
        indexes = list(data.index.values)

    total_indexes = len(indexes)
    np.random.shuffle(indexes)

    batch_sizes = []
    while(True):
        if CONST_BATCH_SIZE is not None:
            size = CONST_BATCH_SIZE
        else:
            size = (sampler.sample() + OFFSET) * SAMPLE_BATCH_SIZE
        if size + np.sum(batch_sizes) <= total_indexes: 
            batch_sizes.append(size)
        else: 
            remanining_batch = (total_indexes - (np.sum(batch_sizes))) // SAMPLE_BATCH_SIZE
            if remanining_batch != 0 and remanining_batch >= (SAMPLE_BATCH_SIZE * OFFSET):
                batch_sizes.append(remanining_batch * SAMPLE_BATCH_SIZE)
            break

    BATCH_SIZES_LIST.extend(batch_sizes)
    batch_sizes  = torch.as_tensor(batch_sizes).reshape(-1,1)
    cumsum = torch.cumsum(torch.as_tensor(batch_sizes),dim = 0).reshape(-1,1)
    if "1.7" in torch.__version__: start_end_indexes = torch.cat((batch_sizes, cumsum), dim = 1)
    else: start_end_indexes = torch.concat((batch_sizes, cumsum), dim = 1)
    start_end_indexes[:,0] = start_end_indexes[:,1] - start_end_indexes[:,0]
    return start_end_indexes

class StratifiedBatchSampler:
    """Stratified batch sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, y, batch_size, shuffle=True):
        if torch.is_tensor(y):
            y = y.numpy()
        assert len(y.shape) == 1, 'label array must be 1D'
        self.n_batches = int(len(y) / batch_size)
        self.skf = StratifiedKFold(n_splits=self.n_batches, shuffle=shuffle)
        self.X = torch.randn(len(y),1).numpy()
        self.y = y
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = torch.randint(0,int(1e8),size=()).item()
        for train_idx, test_idx in self.skf.split(self.X, self.y):
            yield test_idx

    def __len__(self):
        return len(self.y)

class WeightedDataShapleyDataset(Dataset):
    """Weighted Data Shapley dataset."""

    def __init__(self, data_x, data_y, start_end_indexes):
        self.data_x = data_x
        self.data_y = data_y
        self.start_end_indexes = start_end_indexes
        self.transform = get_transforms()
        self.nrows = len(self.data_x)
        self.classidxs = None

    def __len__(self):
        return len(self.start_end_indexes)

    def __getitem__(self, idx):
        
        batch_size = self.start_end_indexes[idx,1] - self.start_end_indexes[idx,0]
        # todo (Error): Very rarely batch_size is in 10K
        if batch_size > (MAX_DATA_SAMPLES_PER_BATCH + OFFSET) * SAMPLE_BATCH_SIZE:
            batch_size = torch.tensor(OFFSET * SAMPLE_BATCH_SIZE)
        if self.classidxs is not None:   
            filtered_idxs = np.where(np.isin(self.data_y, self.classidxs))
        else:
            filtered_idxs = np.arange(len(self.data_y))

        np.random.shuffle(filtered_idxs)
        filtered_y = self.data_y[filtered_idxs]
        filtered_x = self.data_x[filtered_idxs]

        n_batches = len(filtered_idxs[0]) // batch_size
        if n_batches.item() < 2:
            print("Error")
        skf = StratifiedKFold(n_splits=n_batches.item(), shuffle=True)
    
        for i, (_, test_index) in enumerate(skf.split(filtered_x, filtered_y)):
            break
        # _,  = skf.split(self.data_x, self.data_y)        
        x = filtered_x[test_index[:batch_size]]
        y = filtered_y[test_index[:batch_size]]

        assert np.unique(y).shape[0] == len(self.classidxs), "Samplded dataset does not contain all sampled classes"
        assert np.all(y.unique(return_counts=True)[1].numpy() >= 2), "Sampled classes less than 2"
        batched_data = (
            x
            ,y
            # ,self.classidxs
            )


        return batched_data

class CustomTensorDataset(Dataset):
    """Weighted Data Shapley dataset."""

    def __init__(self, data_x, data_y, batch_size):
        self.data_x = data_x
        self.data_y = data_y
        self.nrows = len(self.data_x)
        self.classidxs = None
        self.batch_size = batch_size

    def __len__(self):
        return self.nrows

    def __getitem__(self, idx):
        
        if self.classidxs is not None:   
            filtered_idxs = np.where(np.isin(self.data_y, self.classidxs))
        else:
            filtered_idxs = np.arange(len(self.data_y))

        np.random.shuffle(filtered_idxs)
        filtered_y = self.data_y[filtered_idxs].copy()
        filtered_x = self.data_x[filtered_idxs].copy()

        batch_size = self.batch_size
        n_batches = len(filtered_x) // batch_size
        skf = StratifiedKFold(n_splits=n_batches, shuffle=True)
        
        for i, (_, test_index) in enumerate(skf.split(filtered_x, filtered_y)):
            break
        # _,  = skf.split(self.data_x, self.data_y)        
        x = filtered_x[test_index[:batch_size]]
        y = filtered_y[test_index[:batch_size]]

        assert np.all(np.array(list(Counter(y).values())) >= 2), "Sampled classes less than 2"
        batched_data = (
            torch.tensor(x)
            ,torch.LongTensor(y)
            # ,np.unique(y)
            )


        return batched_data


def get_tensor_dataset(data):
    if isinstance(data, tuple):
        x_val, y_val = data
        if isinstance(x_val, np.ndarray):
            x_val = torch.tensor(x_val, dtype=torch.float32)
            y_val = torch.tensor(y_val, dtype=torch.float32)
        val_set = TensorDataset(x_val, y_val)
    else:
        raise ValueError('val_data must be tuple of tensors ')
    return val_set

def get_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),
        #transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

def get_weighted_data_shapley_dataset(data, sampler, CONST_BATCH_SIZE=None):
    if isinstance(data, tuple):
        data_x, data_y  = data
        indexes = get_batch_indexes(data_x, sampler, CONST_BATCH_SIZE)
        data_set = WeightedDataShapleyDataset(data_x, data_y, indexes)
    else:
        raise ValueError('data must be tuple of tensors ')

    return data_set


def test_get_batch_indexes():
    data = torch.rand((256, 100))

    sampler = get_sampler(data,16,1)
    indexes = get_batch_indexes(data, sampler, None)
    dl = DataLoader(TensorDataset(indexes), batch_size=1,
                        shuffle=True, num_workers=0)

    for x in dl:
        print(x)


if __name__ == "__main__":
    test_get_batch_indexes()
    print("-------------------------------")