import time
import uuid
import torch

import numpy as np

from model import XiheNet

from service.utils import session_pool
from service.utils import BaseHttpRouter
from service.payload import payload_processors


model = XiheNet.load_from_checkpoint(f'./model/model.ckpt')
_ = model.eval().cuda()
scripted_model = torch.jit.script(model)

n_scale = torch.Tensor(model.hparams['scale'])
n_min = torch.Tensor(model.hparams['min'])
processor = payload_processors['point_cloud_xihe_optimized']

# f = open('./dist/perf/edge_processing_time.csv', 'w', buffering=1)
# f.write('component,time\n')


class LightingEstimationHTTPHandler(BaseHttpRouter):
    def post(self):
        # t0 = time.time()

        # Get meta info
        sid = uuid.UUID(str(self.request.headers['Session-ID']))
        session = session_pool[sid]

        # Get point cloud
        pc = processor(self.request.body, session['anchors'])
        # session['point_clouds'].append(pc)
        # pc = np.random.rand(1280, 6).astype(np.float32)

        # np.savetxt('./dist/lighting_estimation/point_cloud.txt', pc)
        # np.save('./dist/lighting_estimation/received', pc)

        xyz = np.moveaxis(pc[:, :3], 0, -1)[np.newaxis, ::]
        rgb = np.moveaxis(pc[:, 3:], 0, -1)[np.newaxis, ::]

        xyz = torch.from_numpy(xyz).cuda()
        rgb = torch.from_numpy(rgb).cuda()

        # Inference
        # t1 = time.time()
        p = scripted_model.forward(xyz, rgb).detach().cpu()
        # t2 = time.time()
        p = (p - n_min) / n_scale
        p = p.numpy()
        coefficients = p.reshape((-1))

        coefficients = coefficients.tolist()
        self.json({'ok': True, 'coefficients': coefficients})

        # t3 = time.time()

        # t_inference = t2 - t1
        # t_processing = t3 - t0 - t_inference

        # f.write(f'processing,{t_processing}\n')
        # f.write(f'inference,{t_inference}\n')


lighting_estimation_http_routes = [
    (r"/lighting-estimation/", LightingEstimationHTTPHandler)
]

__all__ = ['lighting_estimation_http_routes']
