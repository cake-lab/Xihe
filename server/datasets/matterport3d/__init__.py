"""Matterport3D dataset abstraction
"""

import json
import glob
import configs

config_path = './datasets/matterport3d/configs'
matterport3d_root = configs.matterport3d_root


class Matterport3DList:
    def __init__(self):
        self.configs = {}

    def get_config(self, scene_id, depth_name):
        if scene_id not in self.configs:
            self.configs[scene_id] = json.load(
                open(f'{config_path}/{scene_id}.json'))

        return self.configs[scene_id][f'{depth_name}.png']
