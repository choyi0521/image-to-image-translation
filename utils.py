import numpy as np


def set_requires_grad(net, requires_grad):
    for param in net.parameters():
        param.requires_grad = requires_grad


def tensor2image(tensor):
    arr = tensor.data.cpu().numpy()
    arr = (np.transpose(arr, (1, 2, 0)) + 1) / 2.0 * 255.0
    arr = arr.astype(np.uint8)
    return arr