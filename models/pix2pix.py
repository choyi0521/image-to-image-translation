import torch.nn as nn
import torch


class Pix2Pix(nn.module):
    def __init__(self, inner_channels):
        super().__init__()
        self.generator = UNetGenerator(3, inner_channels)
        self.discriminator = NLayerDiscriminator(3, inner_channels)

    def generate(self, x):
        return self.generator(x)


class UNetGenerator(nn.module):
    def __init__(self, image_channels=3, inner_channels=64, n_layers=8):
        super().__init__()
        assert n_layers >= 5

        block = UNetSkipConnectionBlock(inner_channels*8, inner_channels*8, 'innermost')
        for _ in range(n_layers-5):
            block = UNetSkipConnectionBlock(inner_channels*8, inner_channels*8, 'middle', block)
        block = UNetSkipConnectionBlock(inner_channels*4, inner_channels*8, 'middle', block)
        block = UNetSkipConnectionBlock(inner_channels*2, inner_channels*4, 'middle', block)
        block = UNetSkipConnectionBlock(inner_channels, inner_channels*2, 'middle', block)
        self.model = UNetSkipConnectionBlock(image_channels, inner_channels, 'outermost', block)

    def forward(self, x):
        return self.model(x)


class UNetSkipConnectionBlock(nn.Module):
    def __init__(self,
                 outer_channels,
                 inner_channels,
                 module_type,
                 submodule=None
                 ):
        super().__init__()
        if module_type not in ['innermost', 'outermost', 'middle']:
            raise Exception('no such module type')

        down_conv = nn.Conv2d(outer_channels, inner_channels, kernel_size=4, stride=2, padding=1, bias=False)
        down_relu = nn.LeakyReLU(0.2, True)
        down_norm = nn.BatchNorm2d(inner_channels)

        up_relu = nn.ReLU(True)
        up_norm = nn.BatchNorm2d(outer_channels)

        self.outermost = module_type == 'outermost'
        if module_type == 'innermost':
            up_conv = nn.ConvTranspose2d(inner_channels, outer_channels, kernel_size=4, stride=2, padding=1, bias=False)
            modules = [down_relu, down_conv, up_relu, up_conv, up_norm]
        elif module_type == 'outermost':
            up_conv = nn.ConvTranspose2d(inner_channels*2, outer_channels, kernel_size=4, stride=2, padding=1)
            modules = [down_conv, submodule, up_relu, up_conv, nn.Tanh()]
        else:
            up_conv = nn.ConvTranspose2d(inner_channels*2, outer_channels, kernel_size=4, stride=2, padding=1, bias=False)
            modules = [down_relu, down_conv, down_norm, submodule, up_relu, up_conv, up_norm]

        self. model = nn.Sequential(*modules)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    def __init__(self, image_channels=3, inner_channels=64, n_layers=3):
        super().__init__()

        modules = [nn.Conv2d(image_channels, inner_channels, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        for i in range(n_layers-1):
            modules += [
                nn.Conv2d(inner_channels*min(2**i, 8), inner_channels*min(2**(i+1), 8), kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(inner_channels*min(2**(i+1), 8)),
                nn.LeakyReLU(0.2, True)
            ]
        modules += [
            nn.Conv2d(inner_channels * min(2 ** (n_layers-1), 8), inner_channels * min(2 ** n_layers, 8), kernel_size=4, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(inner_channels * min(2 ** n_layers, 8)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(inner_channels * min(2 ** n_layers, 8), 1, kernel_size=4, stride=1, padding=1)
        ]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        self.model(x)
