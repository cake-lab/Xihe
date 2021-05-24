"""XiheNet training script
"""
import os
import json
import yaml
import torch
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from model.trainer import XiheNet
from model.utils import slack
from model.utils import train_valid_test_split
from model.utils import ModelSavingCallback

from datasets.xihe.loader import XiheTestDataset
from datasets.xihe.loader import XiheTrainDataset
from datasets.xihe.loader import XiheTrainD10Dataset

from datasets.xihe.loader import XiheFPSTestDataset
from datasets.xihe.loader import XiheFPSTrainDataset
from datasets.xihe.loader import XiheFPSTrainD10Dataset

from datasets.pointar.loader import PointARTestDataset
from datasets.pointar.loader import PointARTrainDataset
from datasets.pointar.loader import PointARTrainD10Dataset


def train_xihenet(debug=False,
                  use_hdr=True,
                  normalize=False,
                  use_traind10=False,
                  loss='mse',
                  dataset='xihe',
                  n_points=1280,
                  num_workers=16,
                  batch_size=32):
    """Train XiheNet model

    Parameters
    ----------
    debug : bool
        Set debugging flag
    use_hdr : bool
        Use HDR SH coefficients data for training
    normalize : bool
        Normalize SH coefficients
    n_points : int
        Number of model input points, default 1280
    num_workers : int
        Number of workers for loading data, default 16
    batch_size : int
        Training batch size
    """

    EXP_NAME = datetime.now().strftime('%Y/%m/%d/%H_%M_%S')

    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        num_gpu = os.environ['CUDA_VISIBLE_DEVICES']
    else:
        num_gpu = -1

    exp_info = {
        'debug': debug,
        'use_hdr': use_hdr,
        'normalize': normalize,
        'n_points': n_points,
        'loss': loss,
        'dataset': dataset,
        'use_traind10': use_traind10,
        'CUDA_VISIBLE_DEVICES': num_gpu
    }

    # Specify dataset
    if dataset == 'xihe':
        TestDataset = XiheTestDataset

        if debug:
            TrainDataset = TestDataset
        else:
            if use_traind10:
                TrainDataset = XiheTrainD10Dataset
            else:
                TrainDataset = XiheTrainDataset
    elif dataset == 'xihe_fps':
        TestDataset = XiheFPSTestDataset

        if debug:
            TrainDataset = TestDataset
        else:
            if use_traind10:
                TrainDataset = XiheFPSTrainD10Dataset
            else:
                TrainDataset = XiheFPSTrainDataset
    elif dataset == 'pointar':
        TestDataset = PointARTestDataset

        if debug:
            TrainDataset = TestDataset
        else:
            if use_traind10:
                TrainDataset = PointARTrainD10Dataset
            else:
                TrainDataset = PointARTrainDataset
    else:
        print('Unrecognized dataset')
        exit()

    # Get loaders ready
    loader_param = {'use_hdr': use_hdr, 'n_points': n_points}
    loaders, scaler = train_valid_test_split(
        TrainDataset, loader_param,
        TestDataset, loader_param,
        normalize=normalize,
        num_workers=num_workers,
        batch_size=batch_size)

    train_loader, valid_loader, test_loader = loaders

    # Get model ready
    model = XiheNet(hparams={
        'n_shc': 27,
        'n_points': n_points,
        'loss': loss,
        'min': torch.from_numpy(scaler.min_) if normalize else torch.zeros((27)),
        'scale': torch.from_numpy(scaler.scale_) if normalize else torch.ones((27))
    })

    # Train
    sample_input = (
        torch.zeros((1, 3, n_points)).float().cuda(),
        torch.zeros((1, 3, n_points)).float().cuda())

    logger = CSVLogger('./dist/logs', EXP_NAME)
    trainer = pl.Trainer(
        gpus=1,
        logger=logger,
        check_val_every_n_epoch=1,
        callbacks=[
            ModelSavingCallback(
                exp_name=EXP_NAME,
                sample_input=sample_input
            ),
            EarlyStopping(monitor=f'valid_shc_{loss}')
        ])

    print('exp_info', json.dumps(exp_info))

    # Start training
    trainer.fit(
        model,
        train_dataloader=train_loader,
        val_dataloaders=[valid_loader, test_loader])

    # Report exp finished
    slack(f'{EXP_NAME} training finished {json.dumps(exp_info)}')

    # Save experiment information
    with open(f'{logger.log_dir}/info.yaml', 'w') as f:
        yaml.safe_dump(exp_info, f)
