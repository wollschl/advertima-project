import os
import torch
import torch.nn as nn
from pathlib import Path
from src.models.layers import ConvLayer

def _base_model(arch, pretrained, device, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = Model()
    if pretrained:
        script_dir = os.path.dirname(__file__)
        root_dir = Path(script_dir).parents[1]
        state_dict = torch.load(str(root_dir) + '/models/'+arch+'.pt', map_location=device)
        model.load_state_dict(state_dict)
    return model

def get_base_model(pretrained, device='cpu'):
    return _base_model('base_model', pretrained, device)

class Model(nn.Module):

    def __init__(self, in_channels=1, n_classes=10, input_dim=32):
        super().__init__()
        self.layers = nn.ModuleList([
            ConvLayer(in_channels, 16, stride=1, kernel_size=3, padding=1),
            ConvLayer(16, 32, stride=1, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvLayer(32, 64, stride=1, kernel_size=3, padding=1),
            ConvLayer(64, 64, stride=1, kernel_size=3, padding=1)
        ])
        self.linear = nn.Linear(int((input_dim / 2 - 2)**2 * 64), n_classes)



    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return self.linear(x.view(x.shape[0],-1))