'''
Reference:
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
'''
import torch
import torch.nn.functional as F
import torch.nn as nn
try:
    from torchvision.models.utils import load_state_dict_from_url
except:
    from torch.hub import load_state_dict_from_url
import torch.fft as fft
import random
import numpy as np
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out





class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None,args=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.args = args
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        assert args is not None, "you should pass args to resnet"
        if 'cifar' in args["dataset"]:
            if args["model_name"] == "memo":
                self.conv1 = nn.Sequential(
                    nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.BatchNorm2d(self.inplanes),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                )
            else:
                self.conv1 = nn.Sequential(
                    nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),                       
                    nn.BatchNorm2d(self.inplanes), 
                    nn.ReLU(inplace=True))
        else:
            if args["init_cls"] == args["increment"]:
                self.conv1 = nn.Sequential(
                    nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.BatchNorm2d(self.inplanes),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                )
            else:
                self.conv1 = nn.Sequential(
                    nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(self.inplanes),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                )
        


        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = 512 * block.expansion
        # self.fc = nn.Linear(512 * block.expansion, num_classes)  # Removed in _forward_impl

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)  # [bs, 64, 32, 32]

        x_1 = self.layer1(x)  # [bs, 128, 32, 32]
        x_2 = self.layer2(x_1)  # [bs, 256, 16, 16]
        x_3 = self.layer3(x_2)  # [bs, 512, 8, 8]
        x_4 = self.layer4(x_3)  # [bs, 512, 4, 4]

        pooled = self.avgpool(x_4)  # [bs, 512, 1, 1]
        features = torch.flatten(pooled, 1)  # [bs, 512]
        # x = self.fc(x)

        return {
            'fmaps': [x_1, x_2, x_3, x_4],
            'features': features
        }

    def forward(self, x, labels=None, aug_mode=False):
        if aug_mode:
            return self.ffr_forward(x, labels, aug_mode)
        return self._forward_impl(x)
    
    def ffr_forward(self, x, labels=None, aug_mode=False):
        x = self.conv1(x)

        aug_idx = random.choice([0,1,2,3])
        layers = [self.layer1, self.layer2, self.layer3, self.layer4]     
        maps = []   
        for idx, layer in enumerate(layers):
            x = layer(x)
            maps.append(x)
            if aug_idx==idx and aug_mode=='freq_noise':
                x = self.magnitude_noise(x, alpha = 1.0, mean = 0, std_dev=0.75)
            # if aug_idx==idx and aug_mode=='freq_dropout':
            #     x = self.magnitude_dropout(x)
            if aug_idx==idx and aug_mode=='freq_mixup':
                x = self.magnitude_mixup(x) 
            # if aug_idx==idx and aug_mode=='spatial_dropout':
            #     res_layers = [layers[i] for i in range(idx+1, len(layers))]
            #     x = self.spatial_dropout(res_layers, x, labels, p= 0.99) 

        pooled = self.avgpool(x)  # [bs, 512, 1, 1]
        features = torch.flatten(pooled, 1)  # [bs, 512]
        # x = self.fc(x)

        return {
            'fmaps': maps,
            'features': features
        }


    def get_features(self, x, magnitude_layer=False):
        x = self.conv1(x)
        if magnitude_layer == 'conv1':
            _, magnitude = self.fft(x, is_shift=True)
        x = self.layer1(x)
        if magnitude_layer == 'layer1':
            _, magnitude = self.fft(x, is_shift=True)
        x = self.layer2(x)
        if magnitude_layer == 'layer2':
            _, magnitude = self.fft(x, is_shift=True)
        x = self.layer3(x)
        if magnitude_layer == 'layer3':
            _, magnitude = self.fft(x, is_shift=True)
        x = self.layer4(x)
        if magnitude_layer == 'layer4':
            _, magnitude = self.fft(x, is_shift=True)

        if magnitude_layer:
            return x, magnitude.mean(dim=1)
        else:
            return x     

    def spatial_dropout(self, res_layers, x, labels, p=0.9):
        # 生成最终特征
        x_bar = x.clone().detach()
        if len(res_layers) > 0:
            with torch.no_grad():
                for idx, layer in enumerate(res_layers):
                    x_bar = layer(x_bar)

        # 随机生成 dropout mask
        b, c, h, w = x.size()  # 获取输入特征图的维度
        mask = torch.rand((b, 1, h, w)).to(x.device)  # 随机生成 mask，尺寸为 (batch_size, 1, height, width)
        mask = (mask < p).float()  # 根据丢弃概率 p 生成二值化 mask
        mask = F.interpolate(mask, size=x.size()[-2:], mode='bilinear', align_corners=False)  # 将 mask 调整为输入尺寸
        mask = mask.expand_as(x)  # 将 mask 扩展到和输入特征图相同的通道数

        # 应用 mask
        return x * mask



        # # ger final features
        # x_bar = x.clone().detach()
        # if len(res_layers) > 0:
        #     with torch.no_grad():
        #         for idx, layer in enumerate(res_layers):
        #             x_bar = layer(x_bar)

        # # generate attention map
        # fc_weights = self.classifier.weight.data  
        # conv_weights = fc_weights.view(fc_weights.size(0), fc_weights.size(1), 1, 1)  #[num_classes, num_channels, 1, 1]
        # logit = F.conv2d(x_bar, conv_weights)  #[batch_size, num_classes, height, width]
        # b, c, h, w = logit.size()

        # norm_logit = torch.zeros((b, h, w))
        # # for i in range(labels.size(0)):
        # #     norm_attn[i] = (-probabilities[i, labels[i]] * torch.log2(probabilities[i, labels[i]] + 1e-12))
        # for i in range(labels.size(0)):
        #     norm_logit[i] = logit[i, labels[i]]
        # norm_logit = norm_logit.view(b, h*w)
        # logit_max  = norm_logit.max(dim=-1)[0].unsqueeze(dim=-1)
        # logit_min  = norm_logit.min(dim=-1)[0].unsqueeze(dim=-1)
        # norm_logit = (norm_logit - logit_min) / (logit_max - logit_min)
        # norm_logit = norm_logit.view(b, h, w).unsqueeze(dim=1)

        # # norm_ent = torch.zeros((b, h, w)).to(self.device)
        # # probabilities = F.softmax(logit, dim=1)
        # # for i in range(labels.size(0)):
        # #     norm_ent[i] = (-probabilities[i, labels[i]] * torch.log2(probabilities[i, labels[i]] + 1e-12))
        # # norm_ent = norm_ent.view(b, h*w)
        # # ent_max  = norm_ent.max(dim=-1)[0].unsqueeze(dim=-1)
        # # ent_min  = norm_ent.min(dim=-1)[0].unsqueeze(dim=-1)
        # # norm_ent = 1 - (norm_ent - ent_min) / (ent_max - ent_min)
        # # norm_ent = norm_ent.view(b, h, w).unsqueeze(dim=1)

        # # norm_attn = (norm_ent - norm_logit).clip(min=0)
        # # attn_max  = norm_attn.max(dim=-1)[0].unsqueeze(dim=-1)
        # # attn_min  = norm_attn.min(dim=-1)[0].unsqueeze(dim=-1)
        # # norm_attn = 1 - (norm_attn - attn_min) / (attn_max - attn_min)
        # # norm_attn = norm_attn.view(b, h, w).unsqueeze(dim=1)
        

        # # generate mask
        # mask = torch.rand(norm_logit.size()).detach() * norm_logit
        # mask = (mask < (1-p)).float()
        # mask = F.interpolate(mask, size=x.size()[-2:], mode='bilinear', align_corners=False)
        # # mask = F.interpolate(mask, size=(7,7), mode='bilinear')
        # mask = mask.expand_as(x)
        # return x*mask

    def fft(self, x, is_shift=False):
        spectrum = fft.fft2(x, dim=(-2, -1))  # 在空间维度上执行2D傅里叶变换
        phase = torch.angle(spectrum)
        magnitude = torch.abs(spectrum)
        if is_shift:
            magnitude = fft.ifftshift(magnitude)
        return phase, magnitude     

    def ifft(self, magnitude, phase):
        reconstructed_spectrum = magnitude * torch.exp(1j * phase)
        reconstructed_x = fft.ifft2(reconstructed_spectrum, dim=(-2, -1)).real
        return reconstructed_x   

    def magnitude_noise(self, x, alpha = 1, mean = 0, std_dev = 0.75, edge_ratio=2/5):
        # extract pahse and manigtude from images by DCT
        phase, magnitude = self.fft(x)
        B, C, H, W = x.shape
        mean = mean
        std_dev = std_dev
        alpha = alpha

        # freq_cut = int(H * edge_ratio)
        # high_freq_mask = torch.zeros_like(magnitude, dtype=torch.bool)
        # high_freq_mask[:, :, freq_cut:-freq_cut, freq_cut:-freq_cut] = 1

        white_noise = torch.normal(mean=mean, std=std_dev, 
                                size=(B, C, H, W)).to(x.device).detach()
        scaled_white_noise = (1 + alpha * white_noise)
        magnitude = scaled_white_noise*magnitude.clip(max=255, min=0)
        # magnitude[high_freq_mask] = scaled_white_noise[high_freq_mask] * magnitude[high_freq_mask].clip(max=255, min=0)

        # reconstruct images
        reconstructed_x = self.ifft(magnitude, phase)
        return reconstructed_x

    def magnitude_dropout(self, x, p=0.97, edge_ratio= 2/5):
        # extract pahse and manigtude from images by DCT
        phase, magnitude = self.fft(x)

        # enhance: dropout
        batch_size, _, height, width = x.size()
        
        freq_cut = int(height * edge_ratio)  # edge ratio controls how much of the image is considered high frequency
        # high_freq_mask = torch.ones_like(magnitude, dtype=torch.bool).to(x.device)
        # high_freq_mask[:, :, freq_cut:-freq_cut, freq_cut:-freq_cut] = 0
        
        mask = torch.ones_like(magnitude).to(x.device)

        drop_mask = torch.rand((batch_size, 1, height, width)).to(x.device)
        drop_mask = (mask < p).float().detach()
        mask[:, :, freq_cut:-freq_cut, freq_cut:-freq_cut] = drop_mask[:, :, freq_cut:-freq_cut, freq_cut:-freq_cut]
        
        dropped_magnitude = mask*magnitude

        # reconstruct images
        reconstructed_x = self.ifft(dropped_magnitude, phase).clip(max=255, min=0)
        return reconstructed_x

    # def magnitude_mixup(self ,x, edge_ratio = 1/5, alpha= 20):
    #     B, C, H, W = x.shape

    #     # extract pahse and manigtude from images by DCT
    #     phase, magnitude = self.fft(x)

    #     # enhance: magnitude mixup

    #     # lam = torch.rand(batch_size).to(x.device).detach()\
    #     #     .unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
    #     index = torch.randperm(B)

    #     magnitude1, magnitude2 = magnitude, magnitude[index]

    #     # define high frequency mask (outside center 1/8 is high freq)
    #     freq_cut = int(H * edge_ratio)
    #     high_freq_mask = torch.ones_like(magnitude1, dtype=torch.bool)
    #     high_freq_mask[:, :, freq_cut:-freq_cut, freq_cut:-freq_cut] = 0

    #     # generate mixing weights
    #     # lams = np.random.beta(alpha, alpha, B)
    #     # lams = torch.tensor(lams).to(x.device).float()
    #     # lams = lams.view(-1, 1, 1, 1).expand(-1, C, H, W)
    #     # lams = torch.where((lams < 0.4) | (lams > 0.6), torch.tensor(0.5).to(x.device).float(), lams)

    #     lams = torch.rand(B).to(x.device).detach()\
    #         .unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)

    #     # mix high frequencies
    #     mixed_high_freq = lams * magnitude1 + (1-lams) * magnitude2

        
    #     # update high frequency parts only
    #     magnitude[high_freq_mask] = mixed_high_freq[high_freq_mask]

    #     # inverse FFT with original phases
    #     recon_x = self.ifft(magnitude, phase).clip(max=255, min=0)

    #     return recon_x


    def magnitude_mixup(self, x, alpha = 20 ):
        # extract pahse and manigtude from images by DCT
        phase, magnitude = self.fft(x)

        # enhance: magnitude mixup
        B, C, H, W = magnitude.shape

        # lams = np.random.beta(alpha, alpha, B)
        # lams = torch.tensor(lams).to(x.device).float()
        # lams = lams.view(-1, 1, 1, 1).expand(-1, C, H, W)
        # lams = torch.where((lams < 0.4) | (lams > 0.6), torch.tensor(0.5).to(x.device).float(), lams)
        lams = torch.rand(B).unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        lams = torch.where((lams < 0.4) | (lams > 0.6), torch.tensor(0.5).float(), lams).to(x.device).detach()

        index = torch.randperm(B).to(x.device).detach()
        mixed_magnitude = lams * magnitude + (1-lams) * magnitude[index]

        # reconstruct images
        reconstructed_x = self.ifft(mixed_magnitude, phase).clip(max=255, min=0)
        return reconstructed_x


    # def magnitude_mixup(self, x):
    #     # extract pahse and manigtude from images by DCT
    #     phase, magnitude = self.fft(x)

    #     # enhance: magnitude mixup
    #     batch_size = x.size(0)
    #     lam = torch.rand(batch_size).to(x.device).detach()\
    #         .unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
    #     index = torch.randperm(batch_size)
    #     mixed_magnitude = lam * magnitude + (1-lam) * magnitude[index]

    #     # reconstruct images
    #     reconstructed_x = self.ifft(mixed_magnitude, phase).clip(max=255, min=0)
    #     return reconstructed_x

    @property
    def last_conv(self):
        if hasattr(self.layer4[-1], 'conv3'):
            return self.layer4[-1].conv3
        else:
            return self.layer4[-1].conv2


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet10(pretrained=False, progress=True, **kwargs):
    """
    For MEMO implementations of ResNet-10
    """
    return _resnet('resnet10', BasicBlock, [1, 1, 1, 1], pretrained, progress,
                   **kwargs)
        
def resnet26(pretrained=False, progress=True, **kwargs):
    """
    For MEMO implementations of ResNet-26
    """
    return _resnet('resnet26', Bottleneck, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
