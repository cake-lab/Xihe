"""Packing generated data into bundles
Input data, i.e. point cloud, will be bundled into hdf5 database
Label data, i.e. SH coefficients, will saved to npz file
"""
import os
import json
import glob
import configs
import importlib
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

import pycuda.driver as drv
from pycuda.compiler import SourceModule

from datasets.xihe.preprocess.utils import fibonacci_sphere

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = str(min(os.getpid() % 4, 2))
importlib.import_module('pycuda.autoinit')

N_POINTS = 1792
anchors_pos = fibonacci_sphere(N_POINTS)
module_src = open('./datasets/xihe/preprocess/cuda/sphere.cu', 'r').read()
module_src = module_src.replace(
    '#define ANCHOR_SIZE 1280',
    f'#define ANCHOR_SIZE {N_POINTS}')
module = SourceModule(module_src)
nn_search = module.get_function("nn_search")


def sphere_points(point_cloud):
    points = np.array(point_cloud[:, 0:3], dtype=np.float32, copy=True)
    colors = np.array(point_cloud[:, 3:6], dtype=np.float32, copy=True)

    base = np.array([len(points), np.finfo(np.float32).max], dtype=np.float32)
    anchor_distance = np.zeros((N_POINTS, 2), dtype=np.float32) + base

    # GPU nearest neighbor search
    nn_search(
        drv.InOut(anchor_distance),
        drv.In(points),
        drv.In(anchors_pos),
        grid=(len(points), 1, 1),
        block=(1024, 1, 1))

    colors_with_base = np.concatenate((colors, [[0, 0, 0]]), axis=0)
    p_idx = anchor_distance[:, 0].astype(np.int32)

    anchor_clr = colors_with_base[p_idx]
    anchor_dst = anchor_distance[:, 1, np.newaxis].astype(np.float32)
    anchor_dst[anchor_dst == np.finfo(np.float32).max] = 0

    res = np.concatenate(
        (anchors_pos, anchor_clr, anchor_dst), axis=-1)

    return res


def runner(args):
    dataset, i = args

    pc = np.load(
        configs.pointar_dataset_path +
        f'/{dataset}/{i}/point_cloud.npz')['point_cloud']
    # Assume virtual object is placed at [0, 0.1, 0]
    pc -= np.array([0, 0.1, 0, 0, 0, 0])

    os.system(f'mkdir -p {configs.xihe_dataset_path}/{dataset}/{i}')
    np.save(
        configs.xihe_dataset_path +
        f'/{dataset}/{i}/{N_POINTS}.npy',
        sphere_points(pc))


def get_package(dataset):
    g = glob.glob(f'{configs.pointar_dataset_path}/{dataset}/*')

    args = [
        (dataset, i)
        for i in range((len(g)))
    ]

    with Pool(25) as _p:
        _ = list(tqdm(_p.imap(runner, args), total=len(args)))


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
        f'{configs.xihe_dataset_path}/package/{dataset}-shc',
        **results)


def pack(dataset, index='all'):
    os.system(f'mkdir -p {configs.xihe_dataset_path}/package')
    os.system(f'mkdir -p {configs.xihe_dataset_path}/{dataset}')

    if index != 'all':
        pc = np.load(
            configs.pointar_dataset_path +
            f'/{dataset}/{index}/point_cloud.npz')['point_cloud']
        pc -= np.array([0, 0.1, 0, 0, 0, 0])

        os.system(f'mkdir -p {configs.xihe_dataset_path}/{dataset}/{index}')
        np.save(
            configs.xihe_dataset_path +
            f'/{dataset}/{index}/{N_POINTS}.npy',
            sphere_points(pc))

    else:
        get_package(dataset)
        pack_sh_coefficients(dataset)
