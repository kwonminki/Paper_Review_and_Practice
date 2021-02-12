import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def ConvBNormRelu2d(
            in_channels, out_channels,
            kernel_size=3, stride=1, padding=1, 
            bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr

         # Contracting path
        self.enc1_1 = ConvBNormRelu2d(in_channels=1, out_channels=64)
        self.enc1_2 = ConvBNormRelu2d(in_channels=64, out_channels=64)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = ConvBNormRelu2d(in_channels=64, out_channels=128)
        self.enc2_2 = ConvBNormRelu2d(in_channels=128, out_channels=128)
        
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = ConvBNormRelu2d(in_channels=128, out_channels=256)
        self.enc3_2 = ConvBNormRelu2d(in_channels=256, out_channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.enc4_1 = ConvBNormRelu2d(in_channels=256, out_channels=512)
        self.enc4_2 = ConvBNormRelu2d(in_channels=512, out_channels=512)
        
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = ConvBNormRelu2d(in_channels=512, out_channels=1024)
        self.enc5_2 = ConvBNormRelu2d(in_channels=1024, out_channels=1024)
        
        self.up_conv4 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = ConvBNormRelu2d(in_channels=1024, out_channels=512)
        self.dec4_1 = ConvBNormRelu2d(in_channels=512, out_channels=512)

        self.up_conv3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = ConvBNormRelu2d(in_channels=512, out_channels=256)
        self.dec3_1 = ConvBNormRelu2d(in_channels=256, out_channels=256)

        self.up_conv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = ConvBNormRelu2d(in_channels=256, out_channels=128)
        self.dec2_1 = ConvBNormRelu2d(in_channels=128, out_channels=128)

        self.up_conv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = ConvBNormRelu2d(in_channels=128, out_channels=64)
        self.dec1_1 = ConvBNormRelu2d(in_channels=64, out_channels=64)

        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)


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

        x = self.fc(dec1_1)

        return x
