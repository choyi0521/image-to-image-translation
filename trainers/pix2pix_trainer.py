from datasets.facades_dataset import FacadesDataset
from core.torch_trainer import TorchTrainer
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
import torch.nn as nn
from models.pix2pix import Pix2Pix
import torch
from utils import set_requires_grad, tensor2image


class Pix2PixTrainer(TorchTrainer):
    def __init__(self, args):
        super().__init__(args)

        # data loader for training, testing set
        self.train_dataloader = DataLoader(
            dataset=FacadesDataset(self.args, dataset_type='train'),
            num_workers=args.workers,
            batch_size=args.batch_size,
            shuffle=True
        )
        self.test_dataloader = DataLoader(
            dataset=FacadesDataset(self.args, dataset_type='test'),
            num_workers=args.workers,
            batch_size=args.batch_size,
            shuffle=False
        )

        # model components
        self.model = Pix2Pix(args.inner_channels, args.dropout).to(self.device)
        self.generator = self.model.generator
        self.discriminator = self.model.discriminator

        # criterions
        self.criterion_gan = GANLoss().to(self.device)
        self.criterion_l1 = nn.L1Loss().to(self.device)
        self.criterion_mse = nn.MSELoss().to(self.device)

        # optimizers
        self.optimizer_g = Adam(self.generator.parameters(), lr=args.lr)
        self.optimizer_d = Adam(self.discriminator.parameters(), lr=args.lr)

        # schedulers
        def lr_lambda(epoch):
            return 1.0-max(0, epoch-args.epochs*(1-args.decay_ratio)) / (args.decay_ratio*args.epochs+1)
        self.scheduler_g = lr_scheduler.LambdaLR(self.optimizer_g, lr_lambda=lr_lambda)
        self.scheduler_d = lr_scheduler.LambdaLR(self.optimizer_d, lr_lambda=lr_lambda)

        # parameter to save
        self.epoch = 0

    def train(self, filepath=None):
        # load training info
        if filepath is not None:
            self._load(filepath)

        while self.epoch < self.args.epochs:
            self.model.train()
            self.epoch += 1
            for iteration, batch in enumerate(self.train_dataloader):
                real_a, real_b = batch[0].to(self.device), batch[1].to(self.device)
                fake_b = self.generator(real_a)

                # update discriminator
                set_requires_grad(self.discriminator, True)
                self.optimizer_d.zero_grad()

                fake_ab = torch.cat((real_a, fake_b), 1)
                pred_fake = self.discriminator(fake_ab.detach())
                loss_fake = self.criterion_gan(pred_fake, False)

                real_ab = torch.cat((real_a, real_b), 1)
                pred_real = self.discriminator(real_ab)
                loss_real = self.criterion_gan(pred_real, True)

                loss_d = (loss_fake + loss_real)/2
                loss_d.backward()
                self.optimizer_d.step()

                # update generator
                set_requires_grad(self.discriminator, False)
                self.optimizer_g.zero_grad()

                fake_ab = torch.cat((real_a, fake_b), 1)
                pred_fake = self.discriminator(fake_ab)
                loss_gan = self.criterion_gan(pred_fake, True)
                loss_l1 = self.criterion_l1(fake_b, real_b) * self.args.lamb

                loss_g = loss_gan + loss_l1
                loss_g.backward()
                self.optimizer_g.step()

                print('Epoch[{0}]({1}/{2} - Loss_D: {3}, Loss_G: {4}, Loss: {5}'.format(
                    self.epoch,
                    iteration+1,
                    len(self.train_dataloader),
                    loss_d.item(),
                    loss_g.item(),
                    loss_d.item()+loss_g.item()
                ))
            self.scheduler_g.step()
            self.scheduler_d.step()

            if self.epoch % self.args.save_freq == 0:
                self._test()
                self._save()

    def _test(self, generate_images=True):
        self.model.eval()
        with torch.no_grad():
            mse = 0.0
            for iteration, batch in enumerate(self.test_dataloader):
                real_a, real_b = batch[0].to(self.device), batch[1].to(self.device)
                pred_b = self.model.generate(real_a)
                mse += self.criterion_mse(pred_b, real_b).item()/len(self.test_dataloader)
                if generate_images:
                    for i in range(pred_b.shape[0]):
                        im = tensor2image(pred_b[i])
                        im.save(self.args.results_dir+'/'+str(self.args.batch_size*iteration+i+1)+'.jpg')
        print('Epoch[{0}] - MSE: {1}'.format(
            self.epoch, mse
        ))

    def _save(self):
        dic = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'scheduler_g_state_dict': self.scheduler_g.state_dict(),
            'scheduler_d_state_dict': self.scheduler_d.state_dict()
        }
        for suffix in [str(self.epoch), 'last']:
            torch.save(dic, self.args.save_dir+'/checkpoint_{0}.pt'.format(suffix))
            print('Saved checkpoint {0}'.format(
                self.args.save_dir+'/checkpoint_{0}.pt'.format(suffix)
            ))

    def _load(self, filepath):
        dic = torch.load(filepath)
        self.epoch = dic['epoch']
        self.model.load_state_dict(dic['model_state_dict'])
        self.optimizer_g.load_state_dict(dic['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(dic['optimizer_d_state_dict'])
        self.scheduler_g.load_state_dict(dic['scheduler_g_state_dict'])
        self.scheduler_d.load_state_dict(dic['scheduler_d_state_dict'])


class GANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))

        # using lsgan
        self.loss = nn.MSELoss()

    def get_target_tensor(self, x, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(x)

    def __call__(self, x, target_is_real):
        target_tensor = self.get_target_tensor(x, target_is_real)
        return self.loss(x, target_tensor)
