'''
Following Beta-Shapley paper we reduce the dimensionality of image datasets to 32 (features from ResNet18, then PCA)
'''

import torch
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
import numpy as np

# Load Fashion-MNIST, MNIST, and CIFAR10 datasets
transform_cifar = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
transform_mnist = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
fmnist_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_mnist)
mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
cifar10_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar)

# Load pre-trained ResNet18 model from PyTorch model zoo
model = torch.hub.load('pytorch/vision:v0.8.2', 'resnet18', pretrained=True)
model.eval()
print('Loaded resnet18 model from torch hub...')

# Extract penultimate layer outputs from ResNet18 model
fmnist_outputs = []
mnist_outputs = []
cifar10_outputs = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

with torch.no_grad():
    for data, _ in fmnist_dataset:
        output = model(data.unsqueeze(0).to(device))
        fmnist_outputs.append(output.squeeze().detach().cpu().numpy())
    print('FMNIST done...')

    for data, _ in mnist_dataset:
        output = model(data.unsqueeze(0).to(device))
        mnist_outputs.append(output.squeeze().detach().cpu().numpy())
    print('MNIST done...')

    for data, _ in cifar10_dataset:
        output = model(data.unsqueeze(0).to(device))
        cifar10_outputs.append(output.squeeze().detach().cpu().numpy())
    print('CIFAR10 done...')

print('Extracted penultimate features from the model...')
fmnist_outputs = np.array(fmnist_outputs)
mnist_outputs = np.array(mnist_outputs)
cifar10_outputs = np.array(cifar10_outputs)

# Fit PCA model and select first 32 principal components
pca_mnist = PCA(n_components=32)
pca_fmnist = PCA(n_components=32)
pca_cifar = PCA(n_components=32)
pca_mnist.fit(mnist_outputs)
pca_fmnist.fit(fmnist_outputs)
pca_cifar.fit(cifar10_outputs)
components_mnist = pca_mnist.components_
components_fmnist = pca_fmnist.components_
components_cifar = pca_cifar.components_
print('PCA components computed...')

# Save first 32 principal components to .npy file
np.save('pca_components_mnist.npy', components_mnist)
np.save('pca_components_fmnist.npy', components_fmnist)
np.save('pca_components_cifar.npy', components_cifar)
print('Finally 32 dimensional features saved ...')
