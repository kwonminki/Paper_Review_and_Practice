import torch
import torch.nn as nn

from layer import ConvBNormRelu2d


class UNet(nn.Module):
    def __init__(self, nch, nker=64, norm="bnorm", learning_type = "plain"): #채널 수, 커널 수, norm 뭐할지(나중에 instance norm 추가하려고), resdual 하게 할지
        super(UNet, self).__init__()

        self.learning_type = learning_type

         # Contracting path
        self.enc1_1 = ConvBNormRelu2d(in_channels=nch, out_channels=nker, norm=norm)
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

        self.fc = nn.Conv2d(in_channels=nker, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)


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
