import torch
import torch.nn as nn

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def ConvLayer(inp, out, ks=3, s=1, p=1):
    return nn.Conv2d(inp,out,kernel_size=ks,stride=s,padding=p)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.neural_net = nn.Sequential (
            ConvLayer(3, 32), nn.ReLU(),
            ConvLayer(32, 64), nn.ReLU(),
            nn.MaxPool2d(2, 2), # Output: (batch_size, 64, x/2, x/2)
            ConvLayer(64, 128), nn.ReLU(),
            ConvLayer(128, 256), nn.ReLU(),
            nn.MaxPool2d(2, 2), # Output: (batch_size, 256, x/4, x/4)
            ConvLayer(256, 512), nn.ReLU(),
            ConvLayer(512, 1024), nn.ReLU(),
            nn.MaxPool2d(2, 2), # Output: (batch_size, 1024, x/8, x/28
            ConvLayer(1024, 2048), nn.ReLU(),
            nn.MaxPool2d(2, 2), # Output: (batch_size, 2048, x/16, x/16)
            nn.Flatten(), # Output: (batch_size, 2048 * 2 * 2) được huấn luyện trên cifar10
            nn.Linear(2048 * 2 * 2, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 10),
    )
    def forward(self, x):
        return self.neural_net(x)
model = CNN().to(device)