import numpy as np
from datetime import datetime

from utils3d import fibonacci_sphere
from utils3d import spherical_to_cartesian


def handle_point_cloud_column_major(payload):
    pc = np.frombuffer(payload, dtype=np.float32)
    pc = pc.reshape((-1, 3, 2))
    pc = pc.transpose((0, 2, 1))
    pc = pc.reshape((-1, 6))

    return pc


def handle_point_cloud_fib_sphere(payload, anchors):
    pc = np.frombuffer(payload, dtype=np.float32)
    pc = pc.reshape((-1, 4))

    # anchors = fibonacci_sphere(len(pc))
    # xyz = pc[:, -1][:, np.newaxis] * anchors
    xyz = anchors
    rgb = pc[:, :3]

    pc = np.concatenate((xyz, rgb), axis=-1)

    return pc


def handle_point_cloud_row_major(payload):
    pc = np.frombuffer(payload, dtype=np.float32)
    pc = pc.reshape((-1, 6))

    return pc


def handle_point_cloud_float4_no_stripe(payload, anchors):
    pc = np.frombuffer(payload, dtype=np.float32)
    pc = pc.reshape((-1, 4))
    pc_distance = pc[:, -1][:, np.newaxis]

    sphere_xyz = anchors * pc_distance
    sphere_clr = pc[:, :3]

    pc = np.concatenate((sphere_xyz, sphere_clr), axis=-1)

    return pc


def handle_point_cloud_xihe_optimized(payload, anchors):
    pc = np.frombuffer(payload, dtype=np.byte)
    pc = pc.reshape((-1, 7))

    pc_index = np.frombuffer(
        np.array(pc[:, :2], copy=True), dtype=np.uint16)
    pc_colors = pc[:, 2:5].astype(
        np.uint8) / np.array([255], dtype=np.float32)
    pc_distance = np.frombuffer(
        np.array(pc[:, 5:], copy=True), dtype=np.float16)

    len_anchor = len(anchors)

    world_xyz = np.zeros((len_anchor, 3), dtype=np.float32)
    world_xyz[pc_index] = anchors[pc_index] * \
        pc_distance[:, np.newaxis]
    sphere_clr = np.zeros((len_anchor, 3), dtype=np.float32)
    sphere_clr[pc_index] = pc_colors

    pc = np.concatenate((world_xyz, sphere_clr), axis=-1)

    return pc


def handle_client_log(payload: bytes):
    str_datetime = f'{datetime.now():%Y-%m-%d-%H-%-M-%S%z}'
    f = open(f'./dump/client_logs/{str_datetime}.txt', 'w')
    f.write(payload.decode('utf-8'))
    f.close()


def handle_point_cloud_spherical_coordinate(payload):
    pc = np.frombuffer(payload, dtype=np.float32)
    pc = pc.reshape((-1, 3, 2))
    pc = pc.transpose((0, 2, 1))
    pc = pc.reshape((-1, 6))

    pc_sph = pc[:, :3]
    pc_clr = pc[:, 3:]
    pc_car = spherical_to_cartesian(pc_sph)

    pc = np.concatenate((pc_car, pc_clr), axis=-1)

    return pc


def handle_rgbd_ar_session(payload):
    pc = np.frombuffer(payload, dtype=np.float32)
    pc = pc.reshape((-1, 256 * 192, 6))

    str_datetime = f'{datetime.now():%Y-%m-%d-%H-%-M-%S%z}'
    np.save(f'./dump/rgbd_ar_session/{str_datetime}', pc)


payload_processors = {
    'client_log': handle_client_log,
    'point_cloud_column_major': handle_point_cloud_column_major,
    'point_cloud_fib_sphere': handle_point_cloud_fib_sphere,
    'point_cloud_row_major': handle_point_cloud_row_major,
    'point_cloud_float4_no_stripe': handle_point_cloud_float4_no_stripe,
    'point_cloud_xihe_optimized': handle_point_cloud_xihe_optimized,
    'point_cloud_spherical_coordinate': handle_point_cloud_spherical_coordinate
}
