import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class AdaIN2d(nn.Module):
    def __init__(self, style_dim, num_features, device = None):
        super().__init__()
        self.device = device
        self.norm = nn.InstanceNorm2d(num_features, affine=False, device=self.device)
        self.fc = nn.Linear(style_dim, num_features*2, device= self.device)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta



class Augmentation(nn.Module):
    def __init__(self, zdim=10, aug_times=5, device=None, original_weight=0.5):
        super(Augmentation, self).__init__()
        self.zdim = zdim
        self.device = device
        self.aug_times = aug_times
        self.original_weight = original_weight

        kernel_sizes = [5, 9, 13, 17] * (self.aug_times // 4 + 1)
        self.adain_layers = nn.ModuleList()
        self.spatial_layers = nn.ModuleList()
        self.spatial_up_layers = nn.ModuleList()

        for i in range(self.aug_times):
            k_size = kernel_sizes[i]
            self.spatial_layers.append(nn.Conv2d(3, 3, k_size).to(self.device))
            self.spatial_up_layers.append(nn.ConvTranspose2d(3, 3, k_size).to(self.device))

        self.adain_layers = nn.ModuleList([
            AdaIN2d(zdim, 3, device=self.device)
            for _ in range(self.aug_times)
        ])

        self.color = nn.Conv2d(3, 3, 1).to(self.device)

        self.distortion_weights = torch.rand(self.aug_times).to(self.device)
        self.distortion_weights = self.distortion_weights * (1 - self.original_weight) / self.distortion_weights.sum()

        self.weight = torch.zeros(self.aug_times + 1).to(self.device)
        self.weight[:self.aug_times] = self.distortion_weights
        self.weight[-1] = original_weight

        self.tran = transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x):
        data = x

        x = x + torch.randn_like(x) * 0.001
        x_c = torch.tanh(F.dropout(self.color(x), p=.2))

        distorted_outputs = []
        distorted_outputs.append(x_c)

        for i in range(self.aug_times - 1):
            x_down = self.spatial_layers[i](x)
            s = torch.randn(len(x_down), self.zdim).to(self.device)
            x_down = self.adain_layers[i](x_down, s)
            x_up = torch.tanh(self.spatial_up_layers[i](x_down))
            distorted_outputs.append(x_up)

        output = sum(w * d for w, d in zip(self.weight[:-1], distorted_outputs))
        output = output + self.weight[-1] * data

        return output