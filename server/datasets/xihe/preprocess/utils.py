import math
import imageio
import numpy as np
from datasets.matterport3d import matterport3d_root


def fibonacci_sphere(samples=1):

    points = np.zeros((samples, 3), dtype=np.float32)
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points[i] = [x, y, z]

    return points


def set_hdf5_dataset(group, dataset_name, data):
    if dataset_name in group.keys():
        group[dataset_name][::] = data
    else:
        group.create_dataset(dataset_name, data=data, compression='gzip')


def map_hdr(channel):
    channel = channel.astype(np.float32)

    mask = channel < 3000

    channel[mask] = channel[mask] * 8e-8
    channel[~mask] = 0.00024 * \
        1.0002 ** (channel[~mask] - 3000)

    return channel


def get_batched_basis_at(normal_batch):
    batch_size = normal_batch.shape[0]
    x, y, z = normal_batch[:, 0], normal_batch[:, 1], normal_batch[:, 2]
    sh_basis = np.zeros((batch_size, 9))

    # band 0
    sh_basis[:, 0] = 0.282095

    # band 1
    sh_basis[:, 1] = 0.488603 * y
    sh_basis[:, 2] = 0.488603 * z
    sh_basis[:, 3] = 0.488603 * x

    # band 2
    sh_basis[:, 4] = 1.092548 * x * y
    sh_basis[:, 5] = 1.092548 * y * z
    sh_basis[:, 6] = 0.315392 * (3 * z * z - 1)
    sh_basis[:, 7] = 1.092548 * x * z
    sh_basis[:, 8] = 0.546274 * (x * x - y * y)

    return sh_basis


def srgb_to_linear(srgb):
    mask = srgb >= 0.04045
    srgb[mask] = ((srgb[mask] + 0.055) / 1.055)**2.4
    srgb[~mask] = srgb[~mask] / 12.92


def load_rgbd_as_tensor(scene_id, color_image_name, downsample_rate):
    color_image = f'{matterport3d_root}'\
        + f'/{scene_id}' \
        + f'/undistorted_color_images' \
        + f'/{color_image_name}.jpg'

    depth_image_name = color_image_name.replace('_i', '_d')
    depth_image = f'{matterport3d_root}'\
        + f'/{scene_id}' \
        + f'/undistorted_depth_images' \
        + f'/{depth_image_name}.png'

    # According to Matterport document, we need to flip up down the image
    org_color_img = imageio.imread(color_image)
    org_depth_img = imageio.imread(depth_image)

    img_color = np.flipud(org_color_img) / 255
    img_depth = np.flipud(org_depth_img) / 4000

    img_color = img_color[::downsample_rate, ::downsample_rate, :]
    img_depth = img_depth[::downsample_rate, ::downsample_rate]

    return img_color, img_depth


def get_euler_rotation_matrix(angels=(0, 0, 0)):
    rx, ry, rz = angels

    R_z = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])

    R_y = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])

    R = R_z @ R_y @ R_x
    R = R.astype(np.float32)

    return R

# https://en.wikipedia.org/wiki/Cube_mapping


def cube_uv_to_xyz(face, uv) -> np.array:
    uvc = 2.0 * uv - 1.0
    ones = np.ones((uv.shape[0]), dtype=np.float32)

    if face == 0:
        # return np.array([1.0, vc, -uc])
        res = np.stack((ones, uvc[:, 1], -1 * uvc[:, 0]), axis=-1)
    elif face == 1:
        # return np.array([-1.0, vc, uc])
        res = np.stack((-1 * ones, uvc[:, 1], uvc[:, 0]), axis=-1)
    elif face == 2:
        # return np.array([uc, 1.0, -vc])
        res = np.stack((uvc[:, 0], ones, -1 * uvc[:, 1]), axis=-1)
    elif face == 3:
        # return np.array([uc, -1.0, vc])
        res = np.stack((uvc[:, 0], -1 * ones, uvc[:, 1]), axis=-1)
    elif face == 4:
        # return np.array([uc, vc, 1.0])
        res = np.stack((uvc[:, 0], uvc[:, 1], ones), axis=-1)
    elif face == 5:
        # return np.array([-uc, vc, -1.0])
        res = np.stack((-1 * uvc[:, 0], uvc[:, 1], -1 * ones), axis=-1)

    res = res.astype(np.float32)
    return res


def cartesian_to_spherical(points):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    r = np.linalg.norm(points, axis=-1)
    phi = np.arctan2(y, x)
    theta = np.arccos(z / r)

    return np.stack([theta, phi, r], axis=-1)


def euler_rotation_xyz(vertices, angels=(0, 0, 0)):
    # Euler XYZ rotation matrix

    rx, ry, rz = angels

    R_z = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])

    R_y = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])

    R = R_z @ R_y @ R_x
    vertices_rotated = np.dot(vertices, R.T)

    return vertices_rotated
