'''
This file contains the architecture of the weighted shapley estimator
'''
import torch
import torch.nn as nn

# takes two inputs of length 'input_dim' each and gives a single scalar as output
class EstimatorNetwork(nn.Module):
    def __init__(self, input_dim, input_dim_mul = 2):
        super(EstimatorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim_mul * input_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x1, x2=None):
        if x2 is None:
            x = x1
        else:
            x = torch.cat([x1, x2], dim=1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze()


if __name__=='__main__':
    # unit test
    d = 100
    model = EstimatorNetwork(input_dim=d)
    x1 = torch.randn(32, d)
    x2 = torch.randn(32, d)
    output = model(x1, x2)
    print(output)
