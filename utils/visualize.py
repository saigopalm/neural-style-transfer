import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

def imshow(tensor, original_size):
    image = tensor.clone().squeeze(0)
    image = ToPILImage()(image)
    image = image.resize(original_size)
    plt.imshow(image)
    plt.axis('off')
    plt.show()