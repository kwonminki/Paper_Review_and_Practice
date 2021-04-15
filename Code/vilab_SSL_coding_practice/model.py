import torch
import torch.nn as nn

from layer import *

#https://arxiv.org/pdf/1511.06434.pdf
#UNSUPERVISEDREPRESENTATIONLEARNINGWITHDEEPCONVOLUTIONALGENERATIVEADVERSARIALNETWORKS
class DCGAN_Generator(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, norm="bnorm"):
        super(DCGAN_Generator, self).__init__()
        #N x in_channels(noise) x 1 x 1
        self.deconv1 = DEConvBNormRelu2d(in_channels, 16*nker, kernel_size=4, stride=1, padding=0, norm=norm, bias=False, relu= "relu")
        #N x nker*16 x 4 x 4
        self.deconv2 = DEConvBNormRelu2d(16*nker, 8*nker, kernel_size=4, stride=2, padding=1, bias=False, norm=norm, relu="leakyrelu", opts=0.2)

        self.deconv3 = DEConvBNormRelu2d(8*nker, 4*nker, kernel_size=4, stride=2, padding=1, bias=False, norm=norm, relu="leakyrelu", opts=0.2)

        self.deconv4 = DEConvBNormRelu2d(4*nker, 2*nker, kernel_size=4, stride=2, padding=1, bias=False, norm=norm, relu="leakyrelu", opts=0.2)

        self.deconv5 = DEConvBNormRelu2d(2*nker, out_channels, kernel_size=4, stride=2, padding=1, bias=False, norm=None, relu=None)


    def forward(self, x):
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)

        x = torch.tanh(x)

        return x


class DCGAN_Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, norm="bnorm"):
        super(DCGAN_Discriminator, self).__init__()

        self.enc1 = ConvBNormRelu2d(in_channels=in_channels, out_channels=nker, kernel_size=4, stride=2, padding=1, bias=False, norm=norm, relu="leakyrelu", opts=0.2)

        self.enc2 = ConvBNormRelu2d(in_channels=nker, out_channels=nker*2, kernel_size=4, stride=2, padding=1, bias=False, norm=norm, relu="leakyrelu", opts=0.2)

        self.enc3 = ConvBNormRelu2d(in_channels=nker*2, out_channels=nker*4, kernel_size=4, stride=2, padding=1, bias=False, norm=norm, relu="leakyrelu", opts=0.2)

        self.enc4 = ConvBNormRelu2d(in_channels=nker*4, out_channels=nker*8, kernel_size=4, stride=2, padding=1, bias=False, norm=norm, relu="leakyrelu", opts=0.2)

        self.enc5 = ConvBNormRelu2d(in_channels=nker*8, out_channels=nker*16, kernel_size=4, stride=2, padding=1, bias=False, norm=None, relu=None)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)

        x = torch.sigmoid(x)

        return x

