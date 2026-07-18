import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    """
    Splits standard convolution into a spatial depthwise filter 
    and a 1x1 pointwise channel combiner to minimize FLOPs.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 1. Depthwise Convolution: Applies a single filter per input channel
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, 
            stride=stride, padding=1, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # 2. Pointwise Convolution: 1x1 convolution to combine the channels
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, 
            stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        return self.relu(x)


class WakeWordTriggerCNN(nn.Module):
    """
    Stage 1: Ultra-lightweight CNN for always-on keyword spotting.
    Takes a 2D Mel-Spectrogram and outputs a binary trigger probability.
    """
    def __init__(self, num_classes=1):
        super().__init__()
        # Input shape: (Batch, 1, n_mels, time_frames)
        
        # Initial standard convolution to extract base edges
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # Edge-optimized feature extraction
        self.features = nn.Sequential(
            DepthwiseSeparableConv(16, 32, stride=2),
            DepthwiseSeparableConv(32, 64, stride=2),
            DepthwiseSeparableConv(64, 64, stride=1)
        )
        
        # Global Average Pooling flattens the spatial/temporal dimensions
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final classification head
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits
        
    def count_parameters(self):
        """
        Diagnostic tool to prove the model fits within strict memory budgets.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
