from PIL import Image
from torchvision import transforms
import torch

def get_loader(image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

def load_image(image_path, image_size):
    image = Image.open(image_path)
    loader = get_loader(image_size)
    image = loader(image).unsqueeze(0)
    return image
