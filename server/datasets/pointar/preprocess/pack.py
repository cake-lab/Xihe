"""Packing generated data into bundles
Input data, i.e. point cloud, will be bundled into hdf5 database
Label data, i.e. SH coefficients, will be bundled into npz files
"""
import os
import json
import glob
import h5py
import configs
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

N_POINTS = 1280


def set_hdf5_dataset(group, dataset_name, data):
    if dataset_name in group.keys():
        group[dataset_name][::] = data
    else:
        group.create_dataset(dataset_name, data=data, compression='gzip')


def uniform_points(point_cloud) -> np.array:
    idx = np.sort(np.random.choice(
        point_cloud.shape[0], N_POINTS, replace=False))

    p = point_cloud[idx]
    p_pos, p_clr = p[:, :3], p[:, 3:6]

    p = np.concatenate((p_pos, p_clr), axis=-1)
    p = p.astype(np.float32)

    return p


def runner(args):
    dataset, i = args

    pc = np.load(
        configs.pointar_dataset_path +
        f'/{dataset}/{i}/point_cloud.npz')['point_cloud']

    # Assume virtual object is placed at [0, 0.1, 0]
    pc -= np.array([0, 0.1, 0, 0, 0, 0])
    u = uniform_points(pc)

    return u


def get_package(dataset):
    g = glob.glob(f'{configs.pointar_dataset_path}/{dataset}/*')

    points_npz = np.zeros((len(g), N_POINTS, 6), dtype=np.float32)

    args = [
        (dataset, i)
        for i in range((len(g)))
    ]

    with Pool(10) as _p:
        result = list(tqdm(_p.imap(runner, args), total=len(args)))

    for i in range(len(result)):
        points_npz[i] = result[i]

    return points_npz


def pack_sh_coefficients(dataset):
    results = {}

    # for dataset in tqdm(['test', 'train']):
    root = f'{configs.pointar_dataset_path}/{dataset}'
    g = glob.glob(f'{root}/*')

    ldr_sh_coefficients = np.zeros((len(g), 9, 3), dtype=np.float32)
    hdr_sh_coefficients = np.zeros((len(g), 9, 3), dtype=np.float32)

    for i in tqdm(range(len(g))):
        ldr_shc = json.load(open(f'{root}/{i}/shc_ldr.json', 'r'))
        hdr_shc = json.load(open(f'{root}/{i}/shc_hdr.json', 'r'))

        ldr_sh_coefficients[i] = np.array(ldr_shc).reshape((-1, 3))
        hdr_sh_coefficients[i] = np.array(hdr_shc).reshape((-1, 3))

    results['ldr'] = ldr_sh_coefficients
    results['hdr'] = hdr_sh_coefficients

    np.savez_compressed(
        f'{configs.pointar_dataset_path}/package/{dataset}-shc',
        **results)


def pack(dataset, index='all'):
    os.system(f'mkdir -p {configs.pointar_dataset_path}/package')

    if index != 'all':
        f = h5py.File(
            configs.pointar_dataset_path +
            f'/package/pointar-dataset-debug.hdf5', 'a')

        pc = np.load(
            configs.pointar_dataset_path +
            f'/{dataset}/{index}/point_cloud.npz')['point_cloud']
        pc -= np.array([0, 0.1, 0, 0, 0, 0])

        g = f.require_group(f'{dataset}_{N_POINTS}')
        set_hdf5_dataset(g, 'uniform', uniform_points(pc))

    else:
        f = h5py.File(
            configs.pointar_dataset_path +
            f'/package/pointar-dataset.hdf5', 'a')

        # g = f.require_group(f'{dataset}_{N_POINTS}')
        # set_hdf5_dataset(g, 'uniform', get_package(dataset))

        pack_sh_coefficients(dataset)
