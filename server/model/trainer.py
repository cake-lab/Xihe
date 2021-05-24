"""Training control and evaluation
"""
import torch
import torch.nn.functional as F

from model.network import PointConvModel
from model.utils import BaseTrainer


class XiheNet(PointConvModel, BaseTrainer):

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def calculate_train_metrics(self, x, y):
        x_xyz, x_rgb = x

        n_scale = torch.Tensor(self.hparams['scale']).cuda()
        n_min = torch.Tensor(self.hparams['min']).cuda()

        source = self.forward(x_xyz, x_rgb)
        target = y

        source_norm = (source - n_min) / n_scale
        target_norm = (target - n_min) / n_scale

        shc_mse_raw = F.mse_loss(source, target)
        shc_mse = F.mse_loss(source_norm, target_norm)

        if self.hparams['loss'] == 'mse':
            return shc_mse_raw, {
                'shc_mse': shc_mse.item()
            }
        elif self.hparams['loss'] == 'rmse':
            loss = torch.sqrt(shc_mse_raw)
            loss_eval = torch.sqrt(shc_mse)
            return loss, {
                'shc_rmse': loss_eval.item()
            }
        else:
            print('Unrecognized loss')
            exit()

    def calculate_valid_metrics(self, x, y):
        x_xyz, x_rgb = x

        n_scale = torch.Tensor(self.hparams['scale']).cuda()
        n_min = torch.Tensor(self.hparams['min']).cuda()

        source = self.forward(x_xyz, x_rgb)
        target = y

        source_norm = (source - n_min) / n_scale
        target_norm = (target - n_min) / n_scale

        shc_mse_raw = F.mse_loss(source, target)
        shc_mse = F.mse_loss(source_norm, target_norm)

        if self.hparams['loss'] == 'mse':
            return shc_mse_raw, {
                'shc_mse': shc_mse.item()
            }
        elif self.hparams['loss'] == 'rmse':
            loss = torch.sqrt(shc_mse_raw)
            loss_eval = torch.sqrt(shc_mse)
            return loss, {
                'shc_rmse': loss_eval.item()
            }
        else:
            print('Unrecognized loss')
            exit()
