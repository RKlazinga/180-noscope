from torch import nn


class ConvUnit(nn.Module):
    """
    Basic combination of convolution, normalisation and activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, *args, **kwargs):
        super().__init__()

        self.steps = [
            nn.Conv2d(in_channels, out_channels, kernel_size, *args, **kwargs),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        ]

        self.steps = nn.Sequential(*self.steps)

    def forward(self, x):
        return self.steps(x)


class PerspectiveNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        mult = 16
        in_size = 128
        conv_layers = 6
        regression_out = 8

        self.layers = [
            ConvUnit(3, mult, 5),
            nn.MaxPool2d(2)
        ]

        for c in range(1, conv_layers):
            self.layers.append(ConvUnit(mult * (2 ** c), mult * (2 ** (c+1)), 4, padding=1))
            self.layers.append(nn.MaxPool2d(2))

        self.layers.append(nn.Flatten())

        total_vars = (in_size / 2**conv_layers)**2 * mult * (2**(conv_layers-1))
        self.layers.append(nn.Linear(total_vars, total_vars // 2))
        self.layers.append(nn.Linear(total_vars // 2, regression_out))

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)
