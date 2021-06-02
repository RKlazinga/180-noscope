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
