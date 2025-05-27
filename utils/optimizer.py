import torch.optim as optim

def get_optimizer(input_img):
    return optim.LBFGS([input_img.requires_grad_()])
