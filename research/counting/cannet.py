"""
Implementation of Context-Aware Crowd Counting (Accepted to CVPR'19)
http://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Context-Aware_Crowd_Counting_CVPR_2019_paper.pdf
"""
import torch.nn as nn
import torch
from torchvision import models
import collections
import torch.nn.functional as F


class CANNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CANNet, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=1024, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.conv1_1 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv1_2 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv2_1 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv2_2 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv3_1 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv3_2 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv6_1 = nn.Conv2d(512, 512, kernel_size=1, bias=False)
        self.conv6_2 = nn.Conv2d(512, 512, kernel_size=1, bias=False)

        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            fsd = collections.OrderedDict()
            for i in range(len(self.frontend.state_dict().items())):
                temp_key = list(self.frontend.state_dict().items())[i][0]
                fsd[temp_key] = list(mod.state_dict().items())[i][1]
            self.frontend.load_state_dict(fsd)

    def forward(self, x):
        fv = self.frontend(x)
        # S=1
        ave1 = F.adaptive_avg_pool2d(fv, (1, 1))
        ave1 = self.conv1_1(ave1)
        #        ave1=F.relu(ave1)
        s1 = F.upsample(ave1, size=(fv.shape[2], fv.shape[3]), mode='bilinear')
        c1 = s1 - fv
        w1 = self.conv1_2(c1)
        w1 = F.sigmoid(w1)
        # S=2
        ave2 = F.adaptive_avg_pool2d(fv, (2, 2))
        ave2 = self.conv2_1(ave2)
        #        ave2=F.relu(ave2)
        s2 = F.upsample(ave2, size=(fv.shape[2], fv.shape[3]), mode='bilinear')
        c2 = s2 - fv
        w2 = self.conv2_2(c2)
        w2 = F.sigmoid(w2)
        # S=3
        ave3 = F.adaptive_avg_pool2d(fv, (3, 3))
        ave3 = self.conv3_1(ave3)
        #        ave3=F.relu(ave3)
        s3 = F.upsample(ave3, size=(fv.shape[2], fv.shape[3]), mode='bilinear')
        c3 = s3 - fv
        w3 = self.conv3_2(c3)
        w3 = F.sigmoid(w3)
        # S=6
        #        print('fv',fv.mean())
        ave6 = F.adaptive_avg_pool2d(fv, (6, 6))
        #        print('ave6',ave6.mean())
        ave6 = self.conv6_1(ave6)
        #        print(ave6.mean())
        #        ave6=F.relu(ave6)
        s6 = F.upsample(ave6, size=(fv.shape[2], fv.shape[3]), mode='bilinear')
        #        print('s6',s6.mean(),'s1',s1.mean(),'s2',s2.mean(),'s3',s3.mean())
        c6 = s6 - fv
        #        print('c6',c6.mean())
        w6 = self.conv6_2(c6)
        w6 = F.sigmoid(w6)
        #        print('w6',w6.mean())

        fi = (w1 * s1 + w2 * s2 + w3 * s3 + w6 * s6) / (w1 + w2 + w3 + w6 + 0.000000000001)
        #        print('fi',fi.mean())
        #        fi=fv
        x = torch.cat((fv, fi), 1)

        x = self.backend(x)
        x = self.output_layer(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)
