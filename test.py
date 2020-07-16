from torch.utils.data import Dataset
from os import listdir
from PIL import Image
from torchvision import transforms
import numpy as np
from utils import tensor2image

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))
])
img = Image.open('./resources/edges2shoes/val/1_AB.jpg').convert('RGB')
img = transform(img)
tensor2image(img).save('./test.jpg')