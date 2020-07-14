from datasets.edges2shoes_dataset import Edges2ShoesDataset
from core.torch_trainer import TorchTrainer
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from models.pix2pix import Pix2Pix


class Pix2PixTrainer(TorchTrainer):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # data loader for training, testing set
        self.train_dataloader = DataLoader(
            dataset=Edges2ShoesDataset(self.args, is_test=False),
            num_workers=args.workers,
            batch_size=args.batch_size,
            shuffle=True
        )
        self.test_dataset = DataLoader(
            dataset=Edges2ShoesDataset(self.args, is_test=True),
            num_workers=args.workers,
            batch_size=args.batch_size,
            shuffle=False
        )

        # model components
        model = Pix2Pix(args.inner_channels).to(self.device)
        self.generator = model.generator
        self.discriminator = model.discriminator

        # criterions
        self.criterion_gan = GANLoss().to(self.device)
        self.criterion_l1 = nn.L1Loss().to(self.device)
        self.criterion_mse = nn.MSELoss().to(self.device)

        # optimizers
        self.optimizer_g = Adam(self.generator.parameters(), lr=args.lr)
        self.optimizer_d = Adam(self.discriminator.parameters(), lr=args.lr)

    def train(self):
        pass

class GANLoss(nn.Module):
    def __init__(self):
        super().__init__()