import os
import glob
import configs
import numpy as np
from tqdm import tqdm

import torch
from torch_cluster import fps


def process(dataset, index, n_points):
    pc = np.load(
        configs.pointar_dataset_path +
        f'/{dataset}/{index}/point_cloud.npz')['point_cloud']
    pc -= np.array([0, 0.1, 0, 0, 0, 0])

    xyz = pc[:, :3]
    xyz = torch.from_numpy(xyz).cuda()

    fps_idx = fps(xyz, ratio=n_points / pc.shape[0] - 0.000001)
    fps_idx = fps_idx.cpu().numpy()

    pc = pc[fps_idx, :]

    return pc


def sample_fps(dataset, index='all', n_points=1280):
    if index != 'all':
        wd = f'{configs.xihe_fps_dataset_path}/{dataset}/{index}'
        os.system(f'mkdir -p {wd}')

        pc = process(dataset, index, n_points)
        np.save(f'{wd}/{n_points}.npy', pc)

    else:
        g = glob.glob(f'{configs.xihe_dataset_path}/{dataset}/*')

        for i in tqdm(range(len(g))):
            wd = f'{configs.xihe_fps_dataset_path}/{dataset}/{i}'
            os.system(f'mkdir -p {wd}')

            pc = process(dataset, i, n_points)
            np.save(f'{wd}/{n_points}.npy', pc)
