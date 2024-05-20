import torch
import numpy as np
import random

torch.backends.cudnn.enabled = False

def set_seed(seed = 2023):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
K=5
INFERENCE_SAMPLE_BATCH_SIZE = 2048
INFERENCE_TEST_SAMPLE_BATCH_SIZE = 256
SAMPLE_BATCH_SIZE = 32
NUM_CLASSES_PER_BATCH = 3
# MAX_DATA_SAMPLES: While calculating weighted shapley dataset limiting my total dataset size with this number and calculating probs
MAX_DATA_SAMPLES = 1000
# During training, we cant use MAX_DATA_SAMPLES as the batch size bcoz of CUDA error. so limiting the batch sizes  
MAX_DATA_SAMPLES_PER_BATCH = 50 #K * NUM_CLASSES_PER_BATCH * 2
# OFFSET: Minimum samples per batch
OFFSET = 20 #K * NUM_CLASSES_PER_BATCH
BATCH_SIZES_LIST = []

print('Loaded Global Variables!')

id2label = {
    0:"T-Shit/top",
    1:"Trouser",
    2:"Pullover",
    3:"Dress",
    4:"Coat",
    5:"Sandal",
    6:"Shirt",
    7:"Sneaker",
    8:"Bag",
    9:"Ankle Boot"
}

