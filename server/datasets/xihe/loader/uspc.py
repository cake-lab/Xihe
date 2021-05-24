"""Data loader
"""

import configs
import numpy as np

from torch import from_numpy
from torch.utils.data import Dataset

from utils3d.geometry import fibonacci_sphere


class BaseDataset(Dataset):
    n_points: int
    dataset_name: str
    arr_target: np.ndarray
    arr_indices: np.ndarray
    data_root = f'{configs.xihe_dataset_path}/package'

    def __init__(self, n_points=1280, use_hdr=False):
        super(BaseDataset, self).__init__()

        hdr_mark = 'hdr' if use_hdr else 'ldr'

        self.n_points = n_points
        self.anchors = fibonacci_sphere(n_points)
        self.anchors = np.moveaxis(self.anchors, 0, -1)
        self.arr_target = np.moveaxis(self.arr_target[hdr_mark], 1, -1)
        self.arr_indices = np.arange(len(self.arr_target), dtype=np.int)

    def __getitem__(self, idx):
        point_cloud = self.__load_source__(idx)
        target_shc = self.__load_target__(idx)

        point_cloud = np.moveaxis(point_cloud, 0, -1)

        xyz = point_cloud[-1, :] * self.anchors
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
        pc = np.load(
            f'{configs.xihe_dataset_path}/' +
            f'{self.dataset_name}/{self.arr_indices[idx]}/' +
            f'{self.n_points}.npy')
        pc = pc.astype(np.float32)

        return pc

    def __load_target__(self, idx):
        return self.arr_target[self.arr_indices[idx]]


class XiheTrainDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        self.arr_target = np.load(f'{self.data_root}/train-shc.npz')
        super().__init__(*args, **kwargs)
        self.dataset_name = 'train'


class XiheTrainD10Dataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        self.arr_target = np.load(f'{self.data_root}/traind10-shc.npz')
        super().__init__(*args, **kwargs)
        self.dataset_name = 'traind10'


class XiheTestDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        self.arr_target = np.load(f'{self.data_root}/test-shc.npz')
        super().__init__(*args, **kwargs)
        self.dataset_name = 'test'
