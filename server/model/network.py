"""XiheNet model architecture definition
"""

import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from model.pointconv_util import PointConvDensitySetAbstraction


class PointConvModel(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.save_hyperparameters()

        n_points = int(self.hparams['n_points'])

        self.sa1 = PointConvDensitySetAbstraction(
            npoint=n_points, nsample=32, in_channel=3 + 3,
            mlp=[64, 128], bandwidth=0.1, group_all=False
        )

        self.sa2 = PointConvDensitySetAbstraction(
            npoint=1, nsample=0, in_channel=128 + 3,
            mlp=[128, 256], bandwidth=0.2, group_all=True
        )

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.drop3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.drop4 = nn.Dropout(0.2)

        self.fc5 = nn.Linear(64, self.hparams['n_shc'])

    def forward(self, xyz, rgb):
        B, _, _ = xyz.shape
        l1_xyz, l1_points = self.sa1(xyz, rgb)
        _, l2_points = self.sa2(l1_xyz, l1_points)

        x = l2_points.view(B, 256)
        x = self.drop3(F.relu(self.bn3(self.fc3(x))))
        x = self.drop4(F.relu(self.bn4(self.fc4(x))))
        x = self.fc5(x)
        x = F.relu(x)

        return x
