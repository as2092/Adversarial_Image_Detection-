import torch.nn as nn
import torch.nn.functional as F

class ann_net(nn.Module):
    def __init__(self):
        super(ann_net, self).__init__()
        self.fc1 = nn.Linear(3, 20)
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
