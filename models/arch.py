import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SimpleConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return F.relu(self.conv(x) + self.shortcut(x))


class SimpleResNet(nn.Module):
    """
    Mẫu kiến trúc ResNet tùy chỉnh đơn giản.
    Người dùng có thể thêm các lớp model khác trong file này.
    """
    def __init__(self, in_channels=3, num_classes=10, dropout_rate=0.2, **kwargs):
        super(SimpleResNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.layer1 = SimpleConvBlock(32, 64, stride=2)
        self.layer2 = SimpleConvBlock(64, 128, stride=2)
        self.layer3 = SimpleConvBlock(128, 256, stride=2)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        out = self.fc(x)
        return out


class CustomConvNet(nn.Module):
    """
    Mô hình Convolutional Neural Network 4 lớp đơn giản.
    """
    def __init__(self, in_channels=3, num_classes=10, dropout_rate=0.2, **kwargs):
        super(CustomConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class CustomMLP(nn.Module):
    """
    Mô hình Multi-Layer Perceptron cho dữ liệu phẳng hoặc Tabular.
    """
    def __init__(self, in_features=784, num_classes=10, hidden_dims=[256, 128], dropout_rate=0.2, **kwargs):
        super(CustomMLP, self).__init__()
        layers = []
        prev_dim = in_features
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, 1)
        return self.net(x)
