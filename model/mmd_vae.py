import torch
import torch.nn as nn
from torch.nn import functional as F


class Encoder(nn.Module):

    def __init__(self, zdims, NC, NEF):
        super(Encoder, self).__init__()

        # Initialize latent dimension and other parameters
        self.zdims = zdims
        self.nc = NC
        self.nef = NEF

        # Initialize encoding blocks
        self.model = nn.Sequential(

            # Conv layer 1
            nn.Conv2d(in_channels=NC, out_channels=NEF, kernel_size=3, stride=2, padding=2),
            nn.LeakyReLU(0.1, inplace=True),

            # Conv layer 2
            nn.Conv2d(in_channels=NEF, out_channels=NEF * 2, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(NEF * 2),
            nn.LeakyReLU(0.1, inplace=True),

            # Conv layer 3
            nn.Conv2d(in_channels=NEF * 2, out_channels=NEF * 4, kernel_size=3, stride=2, padding=2),
            # nn.BatchNorm2d(NEF * 4),
            nn.LeakyReLU(0.1, inplace=True),

            # Conv layer 4
            nn.Conv2d(in_channels=NEF * 4, out_channels=NEF * 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

        )

        # Initialize linear layer
        self.fc = nn.Linear(2048, self.zdims)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 2048)
        z = self.fc(x)
        return z


class Decoder(nn.Module):

    def __init__(self, zdims, NC, NDF):
        super(Decoder, self).__init__()

        # Initialize latent dimension and other parameters
        self.zdims = zdims
        self.nc = NC
        self.ndf = NDF

        # Initialize decoding blocks
        self.model = nn.Sequential(

            # Deconv layer 1
            nn.ConvTranspose2d(in_channels=NDF * 8, out_channels=NDF * 4, kernel_size=(4, 3), stride=(2, 2), padding=(1, 0)),
            # nn.BatchNorm2d(NDF * 8),
            nn.ReLU(inplace=True),

            # Deconv layer 2
            nn.ConvTranspose2d(in_channels=NDF * 4, out_channels=NDF * 2, kernel_size=(3, 4), stride=(2, 2), padding=(2, 2)),
            # nn.BatchNorm2d(NDF * 4),
            nn.ReLU(inplace=True),

            # Deconv layer 3
            nn.ConvTranspose2d(in_channels=NDF * 2, out_channels=NDF, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
            # nn.BatchNorm2d(NDF * 2),
            nn.ReLU(inplace=True),

            # Deconv layer 4
            nn.ConvTranspose2d(in_channels=NDF, out_channels=NC, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2)),
            nn.Sigmoid()

        )

        # Initialize linear layer
        self.fc = nn.Linear(self.zdims, 2048)

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(-1, 32, 64)
        x = torch.unsqueeze(x, dim=0)
        recon_x = self.model(x)
        return recon_x


class MMD_VAE(nn.Module):

    def __init__(self, zdims, NC, NEF, NDF):
        super(MMD_VAE, self).__init__()
        print("Using MMD-VAE Version-2!")
        self.encoder = Encoder(zdims, NC, NEF)
        self.decoder = Decoder(zdims, NC, NDF)

    def forward(self, x):
        z = self.encoder(x)
        recon_x = self.decoder(z)
        return z, recon_x

    def generate_sequences_from_prior(self, sample):
        return self.decoder(sample)

