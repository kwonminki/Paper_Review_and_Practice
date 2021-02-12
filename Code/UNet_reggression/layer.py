import torch.nn as nn

class ConvBNormRelu2d(nn.Module):
    def __init__(
    self, in_channels, out_channels,
    kernel_size=3, stride=1, padding=1, bias=True, norm="bnorm",
    relu="relu", opts = None):
        super().__init__()

        layers = []
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding,
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

