import torch
import torch.nn as nn
import torch.nn.functional as F
from math import *
import numpy as np
import random

# A NN with pytorch
# Policy has two NNs, policy and critic with common convolutional layer
class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        
        # solution
        self.conv = nn.Conv2d(1, 16, kernel_size=2, stride=1, bias=False)
        self.size = 2*2*16
        self.fc = nn.Linear(self.size,32)

        # layers for the policy(action)
        self.fc_action1 = nn.Linear(32, 16)
        self.fc_action2 = nn.Linear(16, 9)
        
        # layers for the critic
        self.fc_value1 = nn.Linear(32, 8)
        self.fc_value2 = nn.Linear(8, 1)
        self.tanh_value = nn.Tanh() #tanh outputs between -1 and 1
        
        
    def forward(self, x):

        y = F.relu(self.conv(x))
        y = y.view(-1, self.size) # view Returns a new tensor with the same data as the self tensor but of a different shape.
        y = F.relu(self.fc(y))
        
        
        # the action head
        a = F.relu(self.fc_action1(y))
        a = self.fc_action2(a) #output of linear action head,just 9 values for 9 possible actions,no probability here
        #print('a: ',a)
        # availability of moves
        avail = (torch.abs(x.squeeze())!=1).type(torch.FloatTensor)
        #avail = avail.view(-1, 9) # view returns a new tensor with the same data as the self tensor but of a different shape.
        avail = avail.reshape(-1, 9) # view returns a new tensor with the same data as the self tensor but of a different shape.
        
        # locations where actions are not possible, we set the prob to zero
        # Note: 1) This is a modified softmax calculation, instead of exp(a), here exp(a-maxa) is calculated to avoid blowup
        # 2) exp(a-maxa) is multiplied with avail to make the value where move not possible to zero forcefully.
        maxa = torch.max(a)
        # subtract off max for numerical stability (avoids blowing up at infinity)
        exp = avail*torch.exp(a-maxa)
        prob = exp/torch.sum(exp)# final modified softmax calculation, probablility of each action
        
        # the value head(critic head)
        value = F.relu(self.fc_value1(y))
        value = self.tanh_value(self.fc_value2(value)) #tanh outputs between -1 and 1
        return prob.view(3,3), value