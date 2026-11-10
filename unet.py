"""Small U-Net. ~2M params at base=32 - trains in minutes on one GPU."""
import torch, torch.nn as nn

def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True))

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        b = base
        self.d1, self.d2, self.d3, self.d4 = block(in_ch, b), block(b, b*2), block(b*2, b*4), block(b*4, b*8)
        self.pool = nn.MaxPool2d(2)
        self.u3 = nn.ConvTranspose2d(b*8, b*4, 2, 2); self.c3 = block(b*8, b*4)
        self.u2 = nn.ConvTranspose2d(b*4, b*2, 2, 2); self.c2 = block(b*4, b*2)
        self.u1 = nn.ConvTranspose2d(b*2, b,   2, 2); self.c1 = block(b*2, b)
        self.out = nn.Conv2d(b, out_ch, 1)

    def forward(self, x):
        d1 = self.d1(x); d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2)); d4 = self.d4(self.pool(d3))
        x = self.c3(torch.cat([self.u3(d4), d3], 1))
        x = self.c2(torch.cat([self.u2(x), d2], 1))
        x = self.c1(torch.cat([self.u1(x), d1], 1))
        return self.out(x)
