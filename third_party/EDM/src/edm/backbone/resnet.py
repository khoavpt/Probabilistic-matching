import torch.nn as nn


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution without padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias
    )


def conv3x3(in_planes, out_planes, stride=1, groups=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=groups,
        bias=bias,
    )


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride), nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        y = x
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.bn2(self.conv2(y))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class ResNet18(nn.Module):
    """
    Fewer channels
    """

    def __init__(self, config=None):
        super().__init__()
        # Config
        block_dims = config["backbone"]["block_dims"]

        # Networks
        self.conv1 = nn.Conv2d(
            1, block_dims[0], kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(block_dims[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(
            BasicBlock, block_dims[0], block_dims[0], stride=1
        )  # 1/2
        self.layer2 = self._make_layer(
            BasicBlock, block_dims[0], block_dims[1], stride=2
        )  # 1/4
        self.layer3 = self._make_layer(
            BasicBlock, block_dims[1], block_dims[2], stride=2
        )  # 1/8
        self.layer4 = self._make_layer(
            BasicBlock, block_dims[2], block_dims[3], stride=2
        )  # 1/16
        self.layer5 = self._make_layer(
            BasicBlock, block_dims[3], block_dims[4], stride=2
        )  # 1/32

        # For fine matching
        self.fine_conv = nn.Sequential(
            self._make_layer(
                BasicBlock, block_dims[2], block_dims[2], stride=1),
            conv1x1(block_dims[2], block_dims[4]),
            nn.BatchNorm2d(block_dims[4]),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, in_dim, out_dim, stride=1):
        layer1 = block(in_dim, out_dim, stride=stride)
        layer2 = block(out_dim, out_dim, stride=1)
        layers = (layer1, layer2)

        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)  # 1/2
        x2 = self.layer2(x1)  # 1/4
        x3 = self.layer3(x2)  # 1/8
        x4 = self.layer4(x3)  # 1/16
        x5 = self.layer5(x4)  # 1/32

        xf = self.fine_conv(x3)  # 1/8

        return [x3, x4, x5, xf]
