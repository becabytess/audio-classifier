import torch
from torch import nn
class AudioClassifier(nn.Module):
    def __init__(self, num_classes=50):
        super(AudioClassifier, self).__init__() 
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(1 ,32 , kernel_size=(3,3) , stride=(1,1),padding=(1,1))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)) 

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)) 

        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)) 
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)

        
        
    def forward(self,x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool4(torch.relu(self.bn4(self.conv4(x))))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x