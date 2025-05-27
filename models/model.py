import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
from models.losses import ContentLoss, StyleLoss

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

def get_style_model(cnn, normalization_mean, normalization_std, style_img, content_img, style_layers, content_layers):
    normalization = Normalization(normalization_mean, normalization_std)
    model = nn.Sequential(normalization)

    content_losses = []
    style_losses = []

    layer_id = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            layer_id += 1
            name = f'conv_{layer_id}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{layer_id}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'avgpool_{layer_id}'
            layer = nn.AvgPool2d(kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{layer_id}'
        else:
            continue

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{layer_id}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target = model(style_img).detach()
            style_loss = StyleLoss(target)
            model.add_module(f"style_loss_{layer_id}", style_loss)
            style_losses.append(style_loss)

    # trim layers after last loss
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:i+1]

    return model, style_losses, content_losses