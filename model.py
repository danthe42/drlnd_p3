import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

# Initialize a layer's weights:
# Those will be random values from a uniform distribution (see the usage of this function, later) in the interval: [ -1/sqrt(layer size), 1/sqrt(layer size) ] 
def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Network(nn.Module):
    def __init__(self, input_dim, hidden_in_dim, hidden_out_dim, output_dim, actor=False):
        super().__init__()

        if actor is True:
            n = "actor"
        else: 
            n = "critic"

        print("Creating {} model using Linear layers in the following configuration: {} -> {} -> {} -> {} ".format( n, input_dim, hidden_in_dim, hidden_out_dim, output_dim ))

        self.fc1 = nn.Linear(input_dim,hidden_in_dim)
        self.fc2 = nn.Linear(hidden_in_dim,hidden_out_dim)
        self.fc3 = nn.Linear(hidden_out_dim,output_dim)
        self.nonlin = f.relu
        self.actor = actor
        self.reset_parameters()

    # initialize model weight using uniform distribution with an interval inversely proportional to the square root of the layer's size 
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        # use a constant interval in case of the output layer
        self.fc3.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, x):
        if self.actor:
            # return an action directly (policy)
            h1 = self.nonlin(self.fc1(x))
            h2 = self.nonlin(self.fc2(h1))
            return torch.tanh(self.fc3(h2))
        
        else:
            # critic network simply outputs a number, the value of the given state and actions
            h1 = self.nonlin(self.fc1(x))
            h2 = self.nonlin(self.fc2(h1))
            h3 = (self.fc3(h2))
            return h3
            