import torch

class LeNet5Classifier(torch.nn.Module):
    def __init__(self, act="tanh"):
        super(LeNet5Classifier, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(1, 6, 1)
        self.pool1 = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.pool2 = torch.nn.MaxPool2d(2)
        self.conv3 = torch.nn.Conv2d(16, 120, 5)
        self.fc1 = torch.nn.Linear(120, 84)
        self.fc2 = torch.nn.Linear(84, 10)
        self.softmax = torch.nn.Softmax()
        self.relu = torch.nn.ReLU()
        
        self.act = self.relu if act=="relu" else torch.tanh
        
    def forward(self, data):
        conv1 = self.pool1(self.act(self.conv1(data)))
        # print(conv1.shape)
        conv2 = self.pool2(self.act(self.conv2(conv1)))
        # print(conv2.shape)
        conv3 = torch.squeeze(self.act(self.conv3(conv2)))
        # print(conv3.shape)
        fc1 = self.act(self.fc1(conv3))
        # print(fc1.shape)
        fc2 = self.fc2(fc1)
        # print(fc2.shape)
        out = self.softmax(fc2)
        
        return out