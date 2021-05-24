"""Data loader
"""

import h5py
import configs
import numpy as np

from torch import from_numpy
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    arr_source: np.ndarray
    arr_target: np.ndarray
    arr_indices: np.ndarray
    data_root = f'{configs.pointar_dataset_path}/package'

    def __init__(self, use_hdr=False):
        super(BaseDataset, self).__init__()

        hdr_mark = 'hdr' if use_hdr else 'ldr'

        self.f_points = h5py.File(
            f'{self.data_root}/pointar-dataset.hdf5', 'r')

        self.arr_target = np.moveaxis(self.arr_target[hdr_mark], 1, -1)
        self.arr_indices = np.arange(len(self.arr_target), dtype=np.int)

    def __getitem__(self, idx):
        point_cloud = self.__load_source__(idx)
        target_shc = self.__load_target__(idx)

        point_cloud = np.moveaxis(point_cloud, 0, -1)

        xyz = point_cloud[:3, :]
        rgb = point_cloud[3:6, :]

        # Original SH coefficients data is channel last
        # change to channel first as PyTorch use it
        target_shc = target_shc.reshape((-1))
        target = from_numpy(target_shc)

        xyz = from_numpy(xyz)
        rgb = from_numpy(rgb)

        return (xyz, rgb), target

    def __len__(self):
        return len(self.arr_indices)

    def __load_source__(self, idx):
        return self.arr_source[self.arr_indices[idx]]

    def __load_target__(self, idx):
        return self.arr_target[self.arr_indices[idx]]


class PointARTrainDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        self.arr_target = np.load(f'{self.data_root}/train-shc.npz')
        super().__init__(*args, **kwargs)
        self.arr_source = self.f_points[f'train_1280']['uniform']


class PointARTestDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        self.arr_target = np.load(f'{self.data_root}/test-shc.npz')
        super().__init__(*args, **kwargs)
        self.arr_source = self.f_points[f'test_1280']['uniform']
