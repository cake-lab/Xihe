"""Neural Illumination dataset abstraction
"""

import zipfile
import configs


class NeuralIlluminationList:
    def __init__(self, dataset_name):
        root = configs.neural_illumination_path

        self.dataset_name = dataset_name
        self.data = open(f'{root}/{dataset_name}list.txt', 'r').readlines()
        self.data = [v.split(',') for v in self.data]

    def __getitem__(self, index):
        # <scene id>, <surface category>, <observation image>, ix, iy, <illumination map>, mx, my
        entry = self.data[index]
        return {
            'scene_id': entry[0],
            'surface_category': entry[1],
            'observation_image': entry[2],
            'ix': entry[3],
            'iy': entry[4],
            'illumination_map': entry[5],
            'mx': entry[6],
            'mx': entry[7]
        }

    def __len__(self):
        return len(self.data)


class NeuralIlluminationZips:
    def __init__(self):
        self.data = {}

    def __getitem__(self, scene_id):
        if scene_id not in self.data:
            self.data[scene_id] = zipfile.ZipFile(
                f'{configs.neural_illumination_path}' +
                f'/illummaps_{scene_id}.zip')

        return self.data[scene_id]
