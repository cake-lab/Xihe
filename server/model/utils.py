"""Model training utilities
"""
import os
import json
import requests
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler


def post_message(message):
    slack(message)


def slack(message):
    print(message)


def train_valid_test_split(
        train_loader_class, train_loader_init_params,
        test_loader_class, test_loader_init_params,
        normalize=False, num_workers=0, batch_size=32):

    train_loader = train_loader_class(**train_loader_init_params)
    valid_loader = train_loader_class(**train_loader_init_params)
    test_loader = test_loader_class(**test_loader_init_params)

    l = len(train_loader)
    indices = np.arange(l, dtype=np.int)
    valid_idx = np.sort(np.random.choice(l, 2500))

    valid_mask = np.zeros((l), dtype=np.bool)
    valid_mask[valid_idx] = True

    train_loader.arr_indices = indices[~valid_mask]
    valid_loader.arr_indices = indices[valid_mask]

    scaler = None

    if normalize:
        scaler = MinMaxScaler()

        t = train_loader.arr_target.reshape((-1, 3 * 9))
        scaler.fit(t[~valid_mask])  # fit only training part

        t = scaler.transform(t)
        train_loader.arr_target = np.array(t, copy=True)
        valid_loader.arr_target = np.array(t, copy=True)

        t = test_loader.arr_target.reshape((-1, 3 * 9))
        test_loader.arr_target = scaler.transform(t)

    train_loader = DataLoader(
        train_loader,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        batch_size=batch_size)

    valid_loader = DataLoader(
        valid_loader,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        batch_size=batch_size)

    test_loader = DataLoader(
        test_loader,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        batch_size=batch_size)

    return (train_loader, valid_loader, test_loader), scaler


class BaseTrainer(pl.LightningModule):

    def __init__(self, *args, **kwargs):
        super(BaseTrainer, self).__init__(*args, **kwargs)

    # Methods for overwrite
    def calculate_train_metrics(self, x, y):
        return 0, {}

    def calculate_valid_metrics(self, x, y):
        return 0, {}

    # Training delegate
    def training_step(self, batch, batch_nb):
        x, y = batch

        loss, metrics = self.calculate_train_metrics(x, y)

        return {
            'loss': loss,
            'metrics': metrics
        }

    def training_epoch_end(self, step_outputs):
        if 'metrics' not in step_outputs[0]:
            return

        metrics_key = step_outputs[0]['metrics'].keys()
        metrics_mean = {f'train_{k}': 0 for k in metrics_key}

        for output in step_outputs:
            for k in metrics_key:
                metrics_mean[f'train_{k}'] += output['metrics'][k]

        for k in metrics_key:
            metrics_mean[f'train_{k}'] /= len(step_outputs)

        self.log_dict(metrics_mean)

    # Validation delegate
    def validation_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch
        loss, metrics = self.calculate_valid_metrics(x, y)

        if dataloader_idx:
            tb_prefix = ['valid', 'test'][dataloader_idx]
        else:
            tb_prefix = 'valid'

        return {
            'val_loss': loss,
            'metrics': metrics,
            'dataset': tb_prefix
        }

    def validation_epoch_end(self, step_loaders_outputs):
        for loader_outputs in step_loaders_outputs:
            output_0 = loader_outputs[0]

            if 'metrics' not in output_0:
                continue

            name = output_0['dataset']
            metrics_key = output_0['metrics'].keys()
            metrics_mean = {f'{name}_{k}': 0 for k in metrics_key}

            for output in loader_outputs:
                for k in metrics_key:
                    metrics_mean[f'{name}_{k}'] += output['metrics'][k]

            for k in metrics_key:
                metrics_mean[f'{name}_{k}'] /= len(loader_outputs)

            self.log_dict(metrics_mean)


class ModelSavingCallback(pl.Callback):
    def __init__(self, exp_name, sample_input):
        self.exp_name = exp_name
        self.sample_input = sample_input

    def on_epoch_end(self, trainer, pl_module):
        dump_path = f'./dist/checkpoints/{self.exp_name}'
        os.system(f'mkdir -p {dump_path}')

        trainer.save_checkpoint(f'{dump_path}/{trainer.current_epoch}.ckpt')
