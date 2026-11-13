from torch import nn
import ai8x


class ECG1DMax78002(nn.Module):
    def __init__(
        self,
        num_classes=4,
        num_channels=1,
        dimensions=(200, 1),
        bias=False,
        **kwargs
    ):
        super().__init__()

        self.conv1 = ai8x.FusedConv1dReLU(
            num_channels, 16, 3,
            stride=1, padding=1, bias=bias, **kwargs
        )

        self.conv2 = ai8x.FusedMaxPoolConv1dReLU(
            16, 32, 3,
            stride=1, padding=0, bias=bias, **kwargs
        )

        self.conv3 = ai8x.FusedMaxPoolConv1dReLU(
            32, 16, 3,
            stride=1, padding=0, bias=bias, **kwargs
        )

        self.fc = ai8x.Linear(
            16 * 47, num_classes,
            bias=bias, wide=True, **kwargs
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def ecg1dmax78002(pretrained=False, **kwargs):
    assert not pretrained
    return ECG1DMax78002(**kwargs)


models = [
    {
        'name': 'ecg1dmax78002',
        'min_input': 1,
        'dim': 1
    }
]
