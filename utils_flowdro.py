import torch
import torch.nn as nn
import torchdiffeq as tdeq
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_flowmodel():
    int_mtd = 'euler'
    odefunc = MNISTAutoencoder()
    model = ODE(odefunc, int_mtd)
    return model.to(device)

class ODE(nn.Module):
    '''
        odefunc can be any function, as long as its forward mapping takes t,x and outputs 'out, -divf'
        where out is the output of the function and divf is the divergence of the function
        and the shape of out is the same as the shape of x.
    '''

    def __init__(self, odefunc, int_mtd='euler'):
        super(ODE, self).__init__()
        self.odefunc = odefunc
        self.int_mtd = int_mtd

    # Using torchdiffeq
    def forward(self, x, reverse=False):
        integration_times = torch.linspace(0, 1, 2).to(device)
        if reverse:
            integration_times = torch.flip(integration_times, [0])
        predz = tdeq.odeint(self.odefunc, (x), integration_times, method=self.int_mtd)
        return predz
  
class MNISTAutoencoder(nn.Module):
    def __init__(self):
        super(MNISTAutoencoder, self).__init__()
        # Encoder with two convolutional layers
        act = nn.ReLU()
        hid1, hid2, hid3 = 128, 256, 512
        self.encoder = nn.Sequential(
            nn.Conv2d(1, hid1, kernel_size=8, stride=2, padding=3),
            act,
            nn.Conv2d(hid1, hid2, kernel_size=6, stride=2, padding=0), 
            act,
            nn.Conv2d(hid2, hid3, kernel_size=5, stride=1, padding=0),
            act,
        )
        # Decoder with two convolutional transpose layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hid3, hid2, kernel_size=5, stride=1, padding=0),
            act,
            nn.ConvTranspose2d(hid2, hid1, kernel_size=6, stride=2, padding=0),
            act,
            nn.ConvTranspose2d(hid1, 1, kernel_size=8, stride=2, padding=3)
        )
    def forward(self, t, x):
        return self.decoder(self.encoder(x))