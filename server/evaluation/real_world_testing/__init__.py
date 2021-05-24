import os
import imageio
import numpy as np
from tqdm import tqdm
from zipfile import ZipFile
from multiprocessing import Pool


def processor(args):
    i, src_file, rec_name = args
    f_rec = ZipFile(src_file, 'r')

    with f_rec.open(f'{i}/color.bytes') as f:
        img = np.frombuffer(f.read(), dtype=np.uint8)

        base = int(np.sqrt(img.shape[0] // 3 // 12))
        width, height = base * 4, base * 3
        img = img.reshape((height, width, 3))

        p = f'./dist/recording/{rec_name}/frames'
        os.system(f'mkdir -p {p}')
        imageio.imsave(f'{p}/{i}_color.png', img)

    with f_rec.open(f'{i}/depth.bytes') as f:
        img = np.frombuffer(f.read(), dtype=np.float32)

        base = int(np.sqrt(img.shape[0] // 12))
        width, height = base * 4, base * 3
        img = img.reshape((height, width, 1))

    return img


def merge_rec(recording_file):
    with ZipFile(recording_file, 'r') as f_rec:
        n_frames = sum([1 for v in f_rec.namelist() if 'color.bytes' in v])

    rec_name = recording_file.split('/')[-1].split('.')[0]
    args = [(i, recording_file, rec_name) for i in range(n_frames)]

    print('Extracting frame data')
    with Pool(32) as _p:
        depth_buffer = list(tqdm(_p.imap(processor, args), total=len(args)))

    depth_max = np.max(depth_buffer)
    depth_min = np.min(depth_buffer)

    print('Saving normalized depth images')
    for i in tqdm(range(len(depth_buffer))):
        d_depth = depth_buffer[i]
        data = (d_depth - depth_min) / depth_max * np.iinfo(np.uint16).max
        data = data.astype(np.uint16)
        imageio.imsave(
            f'./dist/recording/{rec_name}/frames/{i}_depth.png', data)

    os.system(
        f'ffmpeg -framerate 30 -i ./dist/recording/{rec_name}/frames/%d_color.png ' +
        f'-c:v libx264 ./evaluation/real_world_testing/recordings/{rec_name}_color.mp4')
    os.system(
        f'ffmpeg -framerate 30 -i ./dist/recording/{rec_name}/frames/%d_depth.png ' +
        f'-c:v libx264 ./evaluation/real_world_testing/recordings/{rec_name}_depth.mp4')
