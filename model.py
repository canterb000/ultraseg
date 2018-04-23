import torch
import torch.nn as nn
from torch.nn import init

def conv_layer(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
    output = nn.Sequential(
      nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.BatchNorm2d(num_features=out_channels),
      nn.LeakyReLU(0.1),
      nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.BatchNorm2d(num_features=out_channels),
      nn.LeakyReLU(0.1)
    )
    return output

def upsample_layer(in_channels, out_channels):
    output = nn.Sequential(
      nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False),
      nn.ReLU()
    )
    return output


class unet(nn.Module):
    def __init__(self):
       super(unet, self).__init__()
       self.conv1a = conv_layer(1, 64) 
       self.conv2a = conv_layer(64, 128) 
       self.conv3a = conv_layer(128, 256) 
       self.conv4a = conv_layer(256, 512) 
       self.conv5a = conv_layer(512, 1024) 

       self.conv4b = conv_layer(1024, 512)
       self.conv3b = conv_layer(512, 256)
       self.conv2b = conv_layer(256, 128)
       self.conv1b = conv_layer(128, 64)

       self.conv0b  = nn.Sequential(
         nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),
         nn.Sigmoid()
       )

       self.max_pool = nn.MaxPool2d(2)

       self.upsample5 = upsample_layer(1024, 512)
       self.upsample4 = upsample_layer(512, 256) 
       self.upsample3 = upsample_layer(256, 128) 
       self.upsample2 = upsample_layer(128, 64)

       for m in self.modules():
           if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
               if m.bias is not None:
                   init.xavier_normal(m.weight)
                   init.constant(m.bias, 0)

    def forward(self, x):
        conv1_out = self.conv1a(x)
        conv2_out = self.conv2a(self.max_pool(conv1_out))
        conv3_out = self.conv3a(self.max_pool(conv2_out))
        conv4_out = self.conv4a(self.max_pool(conv3_out))
        conv5_out = self.conv5a(self.max_pool(conv4_out))
        conv5b_out = torch.cat((self.upsample5(conv5_out), conv4_out), 1)
        conv4b_out = self.conv4b(conv5b_out)
        conv3b_out = self.conv3b(torch.cat((self.upsample4(conv4b_out), conv3_out), 1))
        conv2b_out = self.conv2b(torch.cat((self.upsample3(conv3b_out), conv2_out), 1))
        conv1b_out = self.conv1b(torch.cat((self.upsample2(conv2b_out), conv1_out), 1))

        conv0b_out = self.conv0b(conv1b_out)

        return conv0b_out
