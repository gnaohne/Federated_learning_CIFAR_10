defaultInputSize = 3 * 32 * 32
defaultOutputSize = 10  # Number of classes

import torch
import torch.nn as nn
import torch.nn.functional as F

class CIFAR10(nn.Module):
    def __init__(self, inputSize=defaultInputSize, numClasses=defaultOutputSize):
        super().__init__()
        self.linear = nn.Linear(inputSize, numClasses)
        self.layers = [
            lambda xb: xb.reshape(-1, self.linear.in_features), # flatten the input
            # self.normalize,
            self.linear,
        ]

    def parameters(self):
        return self.linear.parameters()

    def normalize(self, xb):
        mean = xb.mean(dim=0, keepdim=True)
        std = xb.std(dim=0, keepdim=True)
        return (xb - mean) / std

    def forward(self, xb):
        val = xb
        for layer in self.layers:
            val = layer(val)
        return val

    def state_dict(self):
        return self.linear.state_dict()

    def load_state_dict(self, state_dict):
        self.linear.load_state_dict(state_dict)

    def save(self, path):
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")

    def predict(self, xb):
        output = self.forward(xb)
        probs = F.softmax(output, dim=1)
        maxProbs, preds = torch.max(probs, dim = 1)
        return preds

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    @property
    def full_param(self):
        return self.linear.weight + self.linear.bias.unsqueeze(1)
