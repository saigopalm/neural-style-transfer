import torch
import torch.nn as nn
import torch.nn.functional as F

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, x):
        self.loss = F.mse_loss(x, self.target)
        return x

def gram_matrix(input):
    batch_size, num_channels, width, height = input.size()
    features = input.view(batch_size * num_channels, width * height)
    G = torch.mm(features, features.t())
    return G.div(batch_size * num_channels * width * height)

class StyleLoss(nn.Module):
    def __init__(self, input):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(input).detach()

    def forward(self, x):
        self.loss = F.mse_loss(gram_matrix(x), self.target)
        return x