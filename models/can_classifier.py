import torch

class CANClassifier(torch.nn.Module):
    def __init__(self, feat_dim=32):
        super(CANClassifier, self).__init__()
        self.feat_dim = feat_dim
        
        self.conv1 = torch.nn.Conv2d(1, self.feat_dim, 3, dilation=1)
        self.conv2 = torch.nn.Conv2d(self.feat_dim, self.feat_dim, 3, dilation=2)
        self.conv3 = torch.nn.Conv2d(self.feat_dim, self.feat_dim, 3, dilation=4)
        self.conv4 = torch.nn.Conv2d(self.feat_dim, self.feat_dim, 3, padding=3, dilation=8)
        self.conv5 = torch.nn.Conv2d(self.feat_dim, 10, 3, dilation=1)
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        conv1 = self.relu(self.conv1(x)) 
        # print(conv1.shape)
        conv2 = self.relu(self.conv2(conv1)) 
        # print(conv2.shape)
        conv3 = self.relu(self.conv3(conv2)) 
        # print(conv3.shape)
        conv4 = self.relu(self.conv4(conv3)) 
        # print(conv4.shape)
        conv5 = self.relu(self.conv5(conv4)) 
        out = torch.mean(conv5, dim=(2,3)) # average pooling over each feature dim, out is (B, 10)
        # print(out.shape)
        return out
        