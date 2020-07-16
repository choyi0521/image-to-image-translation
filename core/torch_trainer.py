import torch
import numpy as np
import os
import random


class TorchTrainer(object):
    def __init__(self, args):
        # set random seed
        self.set_random_seed()

        # cuda setting
        self.args = args
        self.device = torch.device("cuda:{0}".format(args.device) if torch.cuda.is_available() else "cpu")

    def train(self, **kwargs):
        pass

    def test(self, **kwargs):
        pass

    def set_random_seed(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

