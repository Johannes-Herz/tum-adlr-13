import itertools

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int,
    ):
        super().__init__()

        padding: int = kernel_size // 2
        self.block: nn.Sequential = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int,
    ):
        super().__init__()

        self.pool: nn.MaxPool2d = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv_block: ConvBlock = ConvBlock(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return self.conv_block(x)


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        skip_channels: int,
        kernel_size: int,
    ):
        super().__init__()

        self.upconv: nn.ConvTranspose2d = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
        )
        self.conv_block: ConvBlock = ConvBlock(
            in_channels=out_channels + skip_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.upconv(x)
        return self.conv_block(torch.cat([x, skip], dim=1))


class UNet(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, widths: list[int], kernel_size: int
    ):
        super().__init__()

        self.stem: ConvBlock = ConvBlock(
            in_channels=in_channels,
            hidden_channels=widths[0],
            out_channels=widths[0],
            kernel_size=kernel_size,
        )
        self.down_blocks: nn.ModuleList = nn.ModuleList(
            [
                DownBlock(
                    in_channels=in_ch,
                    hidden_channels=out_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                )
                for in_ch, out_ch in itertools.pairwise(widths)
            ]
        )
        self.up_blocks: nn.ModuleList = nn.ModuleList(
            [
                UpBlock(
                    in_channels=in_ch,
                    hidden_channels=out_ch,
                    out_channels=out_ch,
                    skip_channels=out_ch,
                    kernel_size=kernel_size,
                )
                for in_ch, out_ch in itertools.pairwise(reversed(widths))
            ]
        )
        self.head: nn.Conv2d = nn.Conv2d(
            in_channels=widths[0],
            out_channels=out_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        skips: list[torch.Tensor] = []
        for down_block in self.down_blocks:
            skips.append(x)
            x = down_block(x)
        for up_block in self.up_blocks:
            x = up_block(x, skip=skips.pop())
        return self.head(x)
