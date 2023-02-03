import torch
import torch.nn as nn

class CLS(nn.Module):
    def __init__(self, sscale=1):
        super(CLS, self).__init__()
        self.conv_block_1 = self._conv_block_1()
        self.conv_block_2 = self._conv_block_2()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.sscale = sscale
    
    class _conv_block_1(torch.nn.Sequential):
        def __init__(self):
            super().__init__()
            self.conv2d_1 = nn.Conv2d(1, 50, kernel_size=5)
            self.bn_1 = nn.BatchNorm2d(50)
            # self.relu_1 = nn.ReLU(True)
            self.maxpool_1 = nn.MaxPool2d(2)

    class _conv_block_2(torch.nn.Sequential):
        def __init__(self):
            super().__init__()
            self.conv2d_1 = nn.Conv2d(50, 20, kernel_size=5)
            self.bn_1 = nn.BatchNorm2d(20)
            # self.relu_1 = nn.ReLU(True)
            self.maxpool_1 = nn.MaxPool2d(2)
    # 320 50
    # 50 10
    class _linear(torch.nn.Sequential):
        def __init__(self, _in, _out):
            super().__init__()
            self.fc = nn.Linear(_in, _out)
            self.bnf = nn.BatchNorm2d(_out)
    
    def conv_1_clamped(self, x):
        x = self.conv_block_1(x)
        _std_1 = torch.std(torch.flatten(x))
        x = torch.clamp(x,  -self.sscale * _std_1, self.sscale * _std_1)
        # x = torch.relu(x)
        return x

    def conv_2_clamped(self, x):
        x = self.conv_block_2(x)
        _std_2 = torch.std(torch.flatten(x))
        x = torch.clamp(x, -self.sscale * _std_2, self.sscale * _std_2)
        # x = torch.relu(x)
        return x
    
    def linear_clamped(self, x):
        x = self.fc1(x)
        _std_3 = torch.std(torch.flatten(x))
        x = torch.clamp(x,-self.sscale * _std_3, self.sscale * _std_3 )
        return x


    def forward(self, x):
        x = torch.relu(self.conv_1_clamped(x))
        x = torch.relu(self.conv_2_clamped(x))
        x = x.reshape(-1, 320)
        x = self.linear_clamped(x)
        _cls = self.fc2(x)
        
        # _cls = self.linear_1(50, 10)(x)

        return _cls