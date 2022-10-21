"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer


class ConvEncoder(BaseNetwork):
    """ Same architecture as the image discriminator """

    def __init__(self, opt):
        super().__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.ngf
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        if opt.crop_size >= 256:
            self.layer6 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))

        self.so = s0 = 4
        self.fc_mu = nn.Linear(ndf * 8 * s0 * s0, 256)
        self.fc_var = nn.Linear(ndf * 8 * s0 * s0, 256)

        self.actvn = nn.LeakyReLU(0.2, False)
        self.opt = opt

    def forward(self, x):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')

        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        if self.opt.crop_size >= 256:
            x = self.layer6(self.actvn(x))
        x = self.actvn(x)

        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        return mu, logvar



class LadderEncoder(BaseNetwork):
    """ Same architecture as the image discriminator """
    def __init__(self, opt):
        super().__init__()
        # ldmk_nc = 1 if opt.DATASETS.connect_landmarks else opt.DATASETS.num_landmarks
        # ldmk_img_nc = 1  if opt.DATASETS.ldmks_visibility is None else 3
        # ldmk_img_nc *= opt.DATASETS.ldmks_temp_window

        # if opt.ARCHITECTURE.netGL in ['Adainwo']:
        #     nif = 3 + 2 * ldmk_img_nc
        # elif opt.ARCHITECTURE.netGL in ['Spadewo']:
        #     nif = 3 + ldmk_img_nc
        # elif opt.ARCHITECTURE.netGL in ['Hourglass', 'AdaInHourglassImage']:
        #     nif = 3 + 2 * ldmk_nc
        # elif opt.ARCHITECTURE.netGL in ['SpadeHourglassImage']:
        #     nif = 3 + opt.DATASETS.label_nc + 2 + 2 * ldmk_img_nc
        # elif opt.ARCHITECTURE.netGL in ['Adain', 'AdainFullResolution']:
        #     nif = 3 + opt.DATASETS.label_nc + 2 * ldmk_img_nc
        # elif opt.ARCHITECTURE.netGL in ['Spade']:
        #     nif = 3 + opt.DATASETS.label_nc + ldmk_img_nc
        # else:
        #     nif = 3 + ldmk_nc
        nif = 5 
        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        # nef = opt.ENCODER.nef
        nef = 64 
        norm_layer = get_nonspade_norm_layer(opt, 'spectralinstance')
        self.layer1 = norm_layer(nn.Conv2d(nif, nef, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(nef * 1, nef * 2, kw, stride=2, padding=pw))
        # self.up_layer2 = norm_layer(nn.Conv2d(nef * 2, nef * 2, kw, stride=1, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(nef * 2, nef * 4, kw, stride=2, padding=pw))
        # self.up_layer3 = nn.Sequential(norm_layer(nn.Conv2d(nef * 4, nef * 2, kw, stride=1, padding=pw)),
                                    #    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.layer4 = norm_layer(nn.Conv2d(nef * 4, nef * 8, kw, stride=2, padding=pw))
        # self.up_layer4 = nn.Sequential(norm_layer(nn.Conv2d(nef * 8, nef * 2, kw, stride=1, padding=pw)),
                                    #   nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True))
        self.layer5 = norm_layer(nn.Conv2d(nef * 8, nef * 8, kw, stride=2, padding=pw))
        # self.up_layer5 = nn.Sequential(norm_layer(nn.Conv2d(nef * 8, nef * 2, kw, stride=1, padding=pw)),
                                    #    nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True))
        # if opt.DATASETS.crop_size >= 256:
        self.layer6 = norm_layer(nn.Conv2d(nef * 8, nef * 8, kw, stride=2, padding=pw))
        # self.up_layer6 = nn.Sequential(norm_layer(nn.Conv2d(nef * 8, nef * 2, kw, stride=1, padding=pw)),
                                        # nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True))

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.actvn = nn.LeakyReLU(0.2, False)
        self.so = s0 = 4
        # self.fc = nn.Linear(nef * 8 * s0 * s0, opt.ARCHITECTURE.z_dim)
        self.fc = nn.Linear(nef * 8 * s0 * s0, 512)
        self.opt = opt

    def forward(self, x):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')

        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        # features = self.up_layer2(x)
        x = self.layer3(self.actvn(x))
        # features = self.up_layer3(x) + features
        x = self.layer4(self.actvn(x))
        # features = self.up_layer4(x) + features
        x = self.layer5(self.actvn(x))
        # features = self.up_layer5(x) + features
        # if self.opt.DATASETS.crop_size >= 256:
        x = self.layer6(self.actvn(x))
        # features = self.up_layer6(x) + features

        x = self.actvn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x 

