import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid
from torchinfo import summary

class Generator(nn.Module):
    def __init__(self, image_size, isColor=True):
        super(Generator, self).__init__()
        self.image_size = image_size
        self.color_channels = 3 if isColor else 1
        self.stem = nn.Linear(
                        in_features = 100,
                        out_features = 4 * 4 * image_size * 16
                    )
        self.stem_norm = nn.BatchNorm2d(image_size * 16)
        self.stem_act = nn.ReLU(True)
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels = image_size * 16,
                out_channels = image_size * 8,
                kernel_size = 4,
                stride = 2,
                padding = 1,
                bias=False
            ),
            nn.BatchNorm2d(image_size * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                in_channels = image_size * 8,
                out_channels = image_size * 4,
                kernel_size = 4,
                stride = 2,
                padding = 1, 
                bias=False
            ),
            nn.BatchNorm2d(image_size * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                in_channels = image_size * 4,
                out_channels = image_size * 2,
                kernel_size = 4,
                stride = 2,
                padding = 1,
                bias=False
            ),
            nn.BatchNorm2d(image_size * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(
                in_channels = image_size * 2, 
                out_channels = self.color_channels,
                kernel_size = 4,
                stride = 2,
                padding = 1,
                bias=False
            ),
            nn.Tanh()
        )

    def forward(self, x):
        stem = self.stem(x)
        stem = torch.reshape(stem, (-1, self.image_size * 16, 4, 4))
        return self.generator(stem)

class Discriminator(nn.Module):
    def __init__(self, image_size, isColor=True):
        super(Discriminator, self).__init__()
        self.image_size = image_size
        self.color_channels = 3 if isColor else 1
        self.discriminator = nn.Sequential(
            nn.Conv2d(
                in_channels = self.color_channels,
                out_channels = image_size,
                kernel_size = 4,
                stride = 2,
                padding = 1,
                bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                in_channels = image_size,
                out_channels = image_size * 2,
                kernel_size = 4,
                stride = 2,
                padding = 1,
                bias=False
            ),
            nn.BatchNorm2d(image_size * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                in_channels = image_size * 2,
                out_channels = image_size * 4,
                kernel_size =  4,
                stride = 2,
                padding = 1,
                bias=False
            ),
            nn.BatchNorm2d(image_size * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                in_channels = image_size * 4,
                out_channels = image_size * 8,
                kernel_size =  4,
                stride = 2,
                padding = 1,
                bias=False
            ),
            nn.BatchNorm2d(image_size * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                in_channels = image_size * 8,
                out_channels = 1,
                kernel_size = 4,
                stride = 1,
                padding = 0,
                bias=False
            ),
            nn.Flatten()
        )

    def forward(self, x):
        return self.discriminator(x)
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

if __name__ == '__main__':
    generator = Generator(image_size = 64)
    discriminator = Discriminator(image_size = 64)
    summary(generator, input_size = (16, 100))
    summary(discriminator, input_size = (16, 3, 64, 64))