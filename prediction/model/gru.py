import torch.nn as nn
from torch.autograd import Variable
import torch


class GRUNet(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, device):
        super(GRUNet, self).__init__()
        
        self.device = device
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).to(self.device)
        
        h_out, _ = self.gru(x, h_0)
        
        h_out = h_out[:, -1, :].to(self.device)
        
        out = self.fc(h_out).to(self.device)
        
        return out