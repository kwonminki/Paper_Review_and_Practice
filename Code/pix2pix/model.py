import torch
import torch.nn as nn

from layer import *


class Pix2Pix_Generator(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, norm="bnorm"):
        super(Pix2Pix_Generator, self).__init__()

        self.enc1 = ConvBNormRelu2d(in_channels, nker, kernel_size=4, stride=2, norm=None, relu="leakyrelu", opts=0.2)
        self.enc2 = ConvBNormRelu2d(nker, 2*nker, kernel_size=4, padding=1, stride=2, norm=norm, relu="leakyrelu", opts=0.2)
        self.enc3 = ConvBNormRelu2d(2*nker, 4*nker, kernel_size=4, padding=1, stride=2, norm=norm, relu="leakyrelu", opts=0.2)
        self.enc4 = ConvBNormRelu2d(4*nker, 8*nker, kernel_size=4, padding=1, stride=2, norm=norm, relu="leakyrelu", opts=0.2)
        self.enc5 = ConvBNormRelu2d(8*nker, 8*nker, kernel_size=4, padding=1, stride=2, norm=norm, relu="leakyrelu", opts=0.2)
        self.enc6 = ConvBNormRelu2d(8*nker, 8*nker, kernel_size=4, padding=1, stride=2, norm=norm, relu="leakyrelu", opts=0.2)
        self.enc7 = ConvBNormRelu2d(8*nker, 8*nker, kernel_size=4, padding=1, stride=2, norm=norm, relu="leakyrelu", opts=0.2)
        self.enc8 = ConvBNormRelu2d(8*nker, 8*nker, kernel_size=4, padding=1, stride=2, norm=norm, relu="leakyrelu", opts=0.2)
        #Unet 기반
        self.dec1 = DEConvBNormRelu2d(8*nker, 8*nker, kernel_size=4, stride=2, relu="relu")
        self.drop1 = nn.Dropout2d(0.5)
        self.dec2 = DEConvBNormRelu2d(2*8*nker, 8*nker, kernel_size=4, stride=2,norm=norm, relu="relu")
        self.drop2 = nn.Dropout2d(0.5)
        self.dec3 = DEConvBNormRelu2d(2*8*nker, 8*nker, kernel_size=4, stride=2,norm=norm, relu="relu")
        self.drop3 = nn.Dropout2d(0.5)
        self.dec4 = DEConvBNormRelu2d(2*8*nker, 8*nker, kernel_size=4, stride=2,norm=norm, relu="relu")
        self.dec5 = DEConvBNormRelu2d(2*8*nker, 4*nker, kernel_size=4, stride=2,norm=norm, relu="relu")
        self.dec6 = DEConvBNormRelu2d(2*4*nker, 2*nker, kernel_size=4, stride=2,norm=norm, relu="relu")
        self.dec7 = DEConvBNormRelu2d(2*2*nker, nker, kernel_size=4, stride=2,norm=norm, relu="relu")
        self.dec8 = DEConvBNormRelu2d(2*nker, out_channels, kernel_size=4, stride=2,norm=None, relu=None)


    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        enc6 = self.enc6(enc5)
        enc7 = self.enc7(enc6)
        enc8 = self.enc8(enc7)

        dec1 = self.dec1(enc8)
        drop1 = self.drop1(dec1)
        #concat
        dec2 = self.dec2(torch.cat((drop1, enc7), dim=1))
        drop2 = self.drop2(dec2)
        dec3 = self.dec3(torch.cat((drop2, enc6), dim=1))
        drop3 = self.drop3(dec3)
        dec4 = self.dec4(torch.cat((drop3, enc5), dim=1))
        dec5 = self.dec5(torch.cat((dec4, enc4), dim=1))
        dec6 = self.dec6(torch.cat((dec5, enc3), dim=1))
        dec7 = self.dec7(torch.cat((dec6, enc2), dim=1))
        dec8 = self.dec8(torch.cat((dec7, enc1), dim=1))

        x = torch.tanh(dec8)

        return x


class pix2pix_Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, norm="bnorm"):
        super(pix2pix_Discriminator, self).__init__()

        self.enc1 = ConvBNormRelu2d(in_channels=in_channels, out_channels=nker, kernel_size=4, stride=2, padding=1, bias=False, norm=None, relu="leakyrelu", opts=0.2)

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



class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, norm="bnorm", learning_type = "plain"): #채널 수, 커널 수, norm 뭐할지(나중에 instance norm 추가하려고), resdual 하게 할지
        super(UNet, self).__init__()

        self.learning_type = learning_type

         # Contracting path
        self.enc1_1 = ConvBNormRelu2d(in_channels=in_channels, out_channels=nker, norm=norm)
        self.enc1_2 = ConvBNormRelu2d(in_channels=nker, out_channels=nker, norm=norm)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = ConvBNormRelu2d(in_channels=nker, out_channels=2*nker, norm=norm)
        self.enc2_2 = ConvBNormRelu2d(in_channels=2*nker, out_channels=2*nker, norm=norm)
        
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = ConvBNormRelu2d(in_channels=2*nker, out_channels=4*nker, norm=norm)
        self.enc3_2 = ConvBNormRelu2d(in_channels=4*nker, out_channels=4*nker, norm=norm)

        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.enc4_1 = ConvBNormRelu2d(in_channels=4*nker, out_channels=8*nker, norm=norm)
        self.enc4_2 = ConvBNormRelu2d(in_channels=8*nker, out_channels=8*nker, norm=norm)
        
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = ConvBNormRelu2d(in_channels=8*nker, out_channels=16*nker, norm=norm)
        self.enc5_2 = ConvBNormRelu2d(in_channels=16*nker, out_channels=16*nker, norm=norm)
        
        self.up_conv4 = nn.ConvTranspose2d(in_channels=16*nker, out_channels=8*nker, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = ConvBNormRelu2d(in_channels=16*nker, out_channels=8*nker, norm=norm)
        self.dec4_1 = ConvBNormRelu2d(in_channels=8*nker, out_channels=8*nker, norm=norm)

        self.up_conv3 = nn.ConvTranspose2d(in_channels=8*nker, out_channels=4*nker, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = ConvBNormRelu2d(in_channels=8*nker, out_channels=4*nker, norm=norm)
        self.dec3_1 = ConvBNormRelu2d(in_channels=4*nker, out_channels=4*nker, norm=norm)

        self.up_conv2 = nn.ConvTranspose2d(in_channels=4*nker, out_channels=2*nker, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = ConvBNormRelu2d(in_channels=4*nker, out_channels=2*nker, norm=norm)
        self.dec2_1 = ConvBNormRelu2d(in_channels=2*nker, out_channels=2*nker, norm=norm)

        self.up_conv1 = nn.ConvTranspose2d(in_channels=2*nker, out_channels=nker, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = ConvBNormRelu2d(in_channels=2*nker, out_channels=nker, norm=norm)
        self.dec1_1 = ConvBNormRelu2d(in_channels=nker, out_channels=nker, norm=norm)

        self.fc = nn.Conv2d(in_channels=nker, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True)


    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)
        enc5_2 = self.enc5_2(enc5_1)

        up_conv4 = self.up_conv4(enc5_2)
        cat4 = torch.cat((enc4_2, up_conv4), dim=1) #dim 0:batch, 1:channel, 2:height, 3:width 
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        up_conv3 = self.up_conv3(dec4_1)
        cat3 = torch.cat((enc3_2, up_conv3), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        up_conv2 = self.up_conv2(dec3_1)
        cat2 = torch.cat((enc2_2, up_conv2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        up_conv1 = self.up_conv1(dec2_1)
        cat1 = torch.cat((enc1_2, up_conv1), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)


        if self.learning_type == "plain":
            x = self.fc(dec1_1)
        elif self.learning_type == "residual":
            x = self.fc(dec1_1) + x

        return x


class ResNet(nn.Module):
    def __init__(
        self, in_channels, out_channels, 
        nker=64, norm="bnorm", 
        learning_type = "plain", n_blocks=16):
        super(ResNet, self).__init__()

        self.learning_type = learning_type

        self.enc = ConvBNormRelu2d(in_channels=in_channels, out_channels=nker, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu="prelu")

        res = []

        for i in range(n_blocks):
            res += [ResBlock(nker, nker, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu="prelu")]

        self.res = nn.Sequential(*res)

        self.dec = ConvBNormRelu2d(nker, nker, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu="prelu")

        self.fc = nn.Conv2d(in_channels=nker, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=True)


    def forward(self, x):
        x0 = x

        x = self.enc(x)
        x = self.res(x)
        x = self.dec(x)

        if self.learning_type == "plain":
            x = self.fc(x)

        elif self.learning_type == "residual":
            x = self.fc(x) + x0
        
        return x



#https://arxiv.org/pdf/1609.04802.pdf
#Photo-Realistic Single Image Super-Resolution Using a Generative AdversarialNetwork
class SRResNet(nn.Module):
    def __init__(
        self, in_channels, out_channels, 
        nker=64, norm="bnorm", 
        learning_type = "plain", n_blocks=16): #위랑 같은데 반복횟수 추가됨.
        super(SRResNet, self).__init__()

        self.learning_type = learning_type

        self.enc = ConvBNormRelu2d(in_channels, nker, 
                                    kernel_size=9, stride=1, 
                                    padding=4, bias=True, 
                                    norm="bnorm", relu="prelu") #padding은 kernel_size/2 (나머지는 버림)
        
        res = []

        for i in range(n_blocks):
            res += [ResBlock(nker, nker, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu="prelu")]

        self.res = nn.Sequential(*res)

        self.dec = ConvBNormRelu2d(nker, nker, 
                                    kernel_size=3, stride=1, 
                                    padding=1, bias=True, 
                                    norm="bnorm", relu=None)

        #여기서 더해줘야함. forward에서

        ps1 = []

        ps1 += [nn.Conv2d(in_channels=nker, out_channels= 4*nker, kernel_size=3, stride=1, padding=1)]
        # ps1 += [nn.functional.pixel_shuffle()]
        ps1 += [PixelShuffle_mine(ry=2, rx=2)]
        ps1 += [nn.PReLU()]
        self.ps1 = nn.Sequential(*ps1)

        ps2 = []
        
        ps2 += [nn.Conv2d(in_channels=nker, out_channels= 4*nker, kernel_size=3, stride=1, padding=1)]
        # ps2 += [nn.functional.pixel_shuffle()]
        ps2 += [PixelShuffle_mine(ry=2, rx=2)]
        ps2 += [nn.PReLU()]
        self.ps2 = nn.Sequential(*ps2)

        self.fc = nn.Conv2d(in_channels=nker, out_channels= out_channels, kernel_size=9, stride=1, padding=4)

    
    def forward(self, x):
        x = self.enc(x)
        x0 = x

        x = self.res(x)

        x = self.dec(x)
        x = x + x0

        x = self.ps1(x)
        x = self.ps2(x)

        x = self.fc(x)

        return x
