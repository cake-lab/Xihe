from __future__ import annotations

import math
import PIL.Image
import numpy as np


class JointPercentageCalculator:
    def __init__(self, anchor_levels, to_cuda=True):
        self.anchor_group = [
            torch.from_numpy(fibonacci_sphere(v).transpose()).cuda()
            for v in anchor_levels]
        
        self.anchor_dist_group = [
            torch.zeros((v), dtype=torch.long).cuda()
            for v in anchor_levels]
        
    def forward(self, points):
        percentage = 0
        pn = torch.linalg.norm(points, dim=-1, keepdims=True)
        pt = points / pn
        
        for a_idx, anchors in enumerate(self.anchor_group):
            idx = torch.argmax(pt @ anchors, dim=-1)

            i = torch.unique(idx)
            percentage += i.shape[0] / anchors.shape[1]
            
        return percentage
    
    
class JointEntropyCalculator:
    def __init__(self, anchor_levels, to_cuda=True):
        self.anchor_group = [
            torch.from_numpy(fibonacci_sphere(v).transpose()).cuda()
            for v in anchor_levels]
        
        self.anchor_dist_group = [
            torch.zeros((v), dtype=torch.long).cuda()
            for v in anchor_levels]
        
    def forward(self, points):
        entropy = 0
        pn = torch.linalg.norm(points, dim=-1, keepdims=True)
        pt = points / pn
        
        for a_idx, anchors in enumerate(self.anchor_group):
            idx = torch.argmax(pt @ anchors, dim=-1)

            i, c = torch.unique(idx, return_counts=True)
            t = self.anchor_dist_group[a_idx]
            t *= 0
            t[i] = c
            anchor_dist_valued = t[t > 0]
            
            p = anchor_dist_valued / anchor_dist_valued.sum()
            entropy += -1 * torch.sum(p * torch.log2(p))
            
        return entropy.item()


def spherical_to_cartesian(points):
    theta = points[:, 0]
    phi = points[:, 1]
    r = points[:, 2]

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return np.stack((x, y, z), axis=-1)


def equirectangular_uv_to_cartesian(uv):
    u, v = uv[:, :, 0], uv[:, :, 1]

    u = (u / np.max(u) - 0.5) * 2  # [-1, 1]
    phi = u * math.radians(180)
    phi = phi.reshape((-1))

    v = v / np.max(v)  # [0, 1]
    v = v * math.radians(180)
    theta = v.reshape((-1))

    r = np.ones_like(theta, dtype=np.float32)

    coord_spherical = np.stack((theta, phi, r), axis=-1)
    coord_cartesian = spherical_to_cartesian(coord_spherical)

    return coord_cartesian


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


class Canvas:
    def __init__(self, width, height, channels=3):
        self.data = np.zeros((height, width, channels), dtype=np.float32)

    def clear(self):
        self.data *= 0

    def to_pil_image(self):
        data_rgb = (self.data.astype(np.float32) + 1) / 2 * 255
        data_rgb = data_rgb.astype(np.uint8)
        return PIL.Image.fromarray(data_rgb, mode='RGB')


def canvas_equirectangular_panorama(height, channels=3):
    c = Canvas(height * 2, height, channels=channels)

    u = np.arange(height * 2, dtype=np.int)
    v = np.arange(height, dtype=np.int)
    uv = np.stack(np.meshgrid(u, v), axis=-1)

    uv_xyz = equirectangular_uv_to_cartesian(uv)
    uv_xyz = euler_rotation_xyz(
        uv_xyz,
        (math.radians(-90), math.radians(0), 0))
    uv_xyz = euler_rotation_xyz(
        uv_xyz,
        (math.radians(0), math.radians(90), 0))

    c = Canvas(height * 2, height)
    c.data = uv_xyz.reshape((height, height * 2, 3))

    return c


class SphericalHarmonics:
    degrees: int
    cef: np.ndarray
    channel_order: str

    def __init__(self, degrees=2, channels=3, channel_order='first'):
        if degrees > 2:
            raise NotImplementedError('Only support degree <= 2')

        self.degrees = degrees
        self.cef = np.zeros(((degrees + 1) ** 2, channels), dtype=np.float32)
        self.channel_order = channel_order

    @property
    def coefficients(self):
        if self.channel_order == 'first':
            return np.moveaxis(self.cef, 0, -1)
        elif self.channel_order == 'last':
            return self.cef
        else:
            raise ValueError('channel_order must be "first" or "last"')

    @coefficients.setter
    def coefficients(self, value):
        self.cef = value

    @staticmethod
    def from_array(sh_coefficients, channel_order='first'):
        if len(sh_coefficients.shape) == 1:
            if channel_order == 'first':
                sh_coefficients = sh_coefficients.reshape((3, -1))
            else:
                sh_coefficients = sh_coefficients.reshape((-1, 3))

        s_0, s_1 = sh_coefficients.shape

        n_channels = s_0 if channel_order == 'first' else s_1
        n_components = s_1 if channel_order == 'first' else s_0

        degrees = int(math.sqrt(n_components)) - 1
        sh = SphericalHarmonics(degrees=degrees)
        sh.coefficients = sh_coefficients
        sh.channel_order = channel_order

        return sh

    @staticmethod
    def from_sphere_points(points, degrees=2):
        sh = SphericalHarmonics(degrees=degrees)
        sh.project_sphere_points(points)

        return sh

    def get_batched_basis_at(self, normal_matrix):
        matrix_length = normal_matrix.shape[0]
        x, y, z = normal_matrix[:, 0], normal_matrix[:, 1], normal_matrix[:, 2]
        sh_basis = np.zeros((matrix_length, 9))

        # degree 0
        if self.degrees >= 0:
            sh_basis[:, 0] = 0.282095

        # degree 1
        if self.degrees >= 1:
            sh_basis[:, 1] = 0.488603 * y
            sh_basis[:, 2] = 0.488603 * z
            sh_basis[:, 3] = 0.488603 * x

        # degree 2
        if self.degrees >= 2:
            sh_basis[:, 4] = 1.092548 * x * y
            sh_basis[:, 5] = 1.092548 * y * z
            sh_basis[:, 6] = 0.315392 * (3 * z * z - 1)
            sh_basis[:, 7] = 1.092548 * x * z
            sh_basis[:, 8] = 0.546274 * (x * x - y * y)

        return sh_basis

    def project_sphere_points(self, sphere_points):
        matrix_sh_basis = self.get_batched_basis_at(
            sphere_points.positions)

        c = np.einsum(
            'ij,ik->ijk',
            matrix_sh_basis,
            sphere_points.features).sum(axis=0)

        norm = (4 * math.pi) / sphere_points.positions.shape[0]
        self.coefficients = c * norm

    def reconstruct(self, canvas_norm: np.ndarray):
        s = canvas_norm.shape

        canvas_norm = canvas_norm.reshape((-1, 3))

        x = canvas_norm[:, 0, np.newaxis]
        y = canvas_norm[:, 1, np.newaxis]
        z = canvas_norm[:, 2, np.newaxis]

        canvas = np.zeros_like(canvas_norm, dtype=np.float32)

        if self.degrees >= 0:
            canvas += self.coefficients[0, :] * 0.886227

        if self.degrees >= 1:
            canvas += self.coefficients[1, :] * 2.0 * 0.511664 * y
            canvas += self.coefficients[2, :] * 2.0 * 0.511664 * z
            canvas += self.coefficients[3, :] * 2.0 * 0.511664 * x

        if self.degrees >= 2:
            canvas += self.coefficients[4, :] * 2.0 * 0.429043 * x * y
            canvas += self.coefficients[5, :] * 2.0 * 0.429043 * y * z
            canvas += self.coefficients[6, :] * 0.743125 * z * z - 0.247708
            canvas += self.coefficients[7, :] * 2.0 * 0.429043 * x * z
            canvas += self.coefficients[8, :] * 0.429043 * (x * x - y * y)

        canvas = canvas.reshape(s)

        return canvas

    def reconstruct_to_canvas(self, canvas=None):
        if canvas is None:
            canvas = canvas_equirectangular_panorama(
                height=128)

        canvas_res = canvas_equirectangular_panorama(
            height=canvas.data.shape[0])
        canvas_res.data = self.reconstruct(canvas.data)

        return canvas_res

    def vis_as_pil_image(self, canvas=None):
        if canvas is None:
            canvas = canvas_equirectangular_panorama(
                height=128)

        arr = self.reconstruct(canvas.data)
        img = PIL.Image.fromarray((arr * 255).astype(np.uint8))

        return img
