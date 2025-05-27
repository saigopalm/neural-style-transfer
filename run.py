import torch
from torchvision.models import vgg19, VGG19_Weights
from utils.image_loader import load_image
from utils.visualize import imshow
from style_transfer import run_style_transfer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_size = (512, 512)
content_img = load_image('images/content.jpg', image_size).to(device)
style_img = load_image('images/style.jpg', image_size).to(device)

input_img = content_img.clone()

cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()

style_layers = ['conv_1', 'conv_3', 'conv_5', 'conv_9', 'conv_13']
content_layers = ['conv_10']

output = run_style_transfer(cnn, content_img, style_img, input_img,
                            style_layers=style_layers,
                            content_layers=content_layers)

imshow(output, original_size=(image_size[0], image_size[1]))
