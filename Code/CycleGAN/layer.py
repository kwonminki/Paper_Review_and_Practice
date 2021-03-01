import torch.nn as nn

class ConvBNormRelu2d(nn.Module):
    def __init__(
        self, in_channels, out_channels,
        kernel_size=3, stride=1, padding=1, bias=True, norm="bnorm",
        relu="relu", opts = None, padding_mode='zeros'):
        super().__init__()

        layers = []
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride,
                                padding=padding, padding_mode=padding_mode,
                                bias=bias)]
        # Norm 설정
        if not norm is None:
            if norm == "bnorm":
                layers += [nn.BatchNorm2d(num_features=out_channels)]
            elif norm == "inorm":
                layers += [nn.InstanceNorm2d(num_features=out_channels)]
        # Relu 설정
        if not relu is None:
            if relu == "relu":
                layers += [nn.ReLU()]
            elif relu == "leakyrelu":
                layers += [nn.LeakyReLU(opts)]
            elif relu == "prelu":
                layers += [nn.PReLU()]

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)


class ResBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels,
        kernel_size=3, stride=1, padding=1, bias=True, norm="bnorm",
        relu="relu", opts = None):
        super().__init__()

        layers = []

        layers += [ConvBNormRelu2d(in_channels, out_channels, 
                                    kernel_size=kernel_size, stride=stride, 
                                    padding=padding, bias=bias,
                                    norm=norm, relu=relu,
                                    opts= opts)]   
        # Norm 설정
        if not norm is None:
            if norm == "bnorm":
                layers += [nn.BatchNorm2d(num_features=out_channels)]
            elif norm == "inorm":
                layers += [nn.InstanceNorm2d(num_features=out_channels)]

        layers += [ConvBNormRelu2d(out_channels, out_channels, 
                                    kernel_size=kernel_size, stride=stride, 
                                    padding=padding, bias=bias,
                                    norm=norm, relu=None,
                                    opts= opts)] #2번째는 relu가 없음.
        
        self.resblock = nn.Sequential(*layers)


    def forward(self, x):
        return x + self.resblock(x)



class PixelShuffle_mine(nn.Module):
    def __init__(self, ry=2, rx=2):
        super().__init__()

        self.ry = ry
        self.rx = rx

    def forward(self, x):
        [Batch, Channel, Height, Width] = list(x.shape)

        x = x.reshape(Batch, Channel//(self.ry * self.rx), self.ry, self.rx, Height, Width)

        x = x.permute(0, 1, 4, 2, 5, 3)

        x = x.reshape(Batch, Channel//(self.ry * self.rx), Height*self.ry, Width*self.rx)

        return x


    
class PixelUnShuffle_mine(nn.Module):
    def __init__(self, ry=2, rx=2):
        super().__init__()

        self.ry = ry
        self.rx = rx

    def forward(self, x):
        [Batch, Channel, Height, Width] = list(x.shape)

        x = x.reshape(Batch, Channel, Height//self.ry, self.ry, Width//self.rx, self.rx)

        x = x.permute(0, 1, 3, 5, 2, 4) #b, c, ry, rx, h/y, w/x

        return x


class DEConvBNormRelu2d(nn.Module):
    def __init__(
        self, in_channels, out_channels,
        kernel_size=3, stride=1, padding=1, bias=True, norm="bnorm",
        relu="relu", opts = None):
        super().__init__()

        layers = []
        layers += [nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride,
                                padding=padding, output_padding=padding,
                                bias=bias)]
        # Norm 설정
        if not norm is None:
            if norm == "bnorm":
                layers += [nn.BatchNorm2d(num_features=out_channels)]
            elif norm == "inorm":
                layers += [nn.InstanceNorm2d(num_features=out_channels)]
        # Relu 설정
        if not relu is None:
            if relu == "relu":
                layers += [nn.ReLU()]
            elif relu == "leakyrelu":
                layers += [nn.LeakyReLU(opts)]
            elif relu == "prelu":
                layers += [nn.PReLU()]

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)