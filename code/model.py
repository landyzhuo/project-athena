import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, size, hidden_size, classes):
        super(NeuralNetwork, self).__init__()
        self.l_1 = nn.Linear(size, hidden_size)
        self.l_2 = nn.Linear(hidden_size, hidden_size)
        self.l_3 = nn.Linear(hidden_size, classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l_1(x)
        out = self.relu(out)
        out = self.l_2(out)
        out = self.relu(out)
        out = self.l_3(out)
        out = self.relu(out)
        return out
