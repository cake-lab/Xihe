import os
import json
import imageio
import numpy as np
from datetime import datetime

from service.utils import BaseHttpRouter


class RecordingHTTPHandler(BaseHttpRouter):
    def post(self):
        t_payload = self.request.headers['Payload-Type']

        if t_payload == 'info':
            archive_name = datetime.now().strftime('%Y/%m/%d/%H_%M_%S')
            p_probe = self.request.headers['Probe-Position']

            os.system(f'mkdir -p ./dist/recording/{archive_name}')
            with open(f'./dist/recording/{archive_name}/info.yaml', 'w') as f:
                f.write(json.dumps({'p_probe': p_probe}))

            self.json({'Ok': True, 'ArchiveName': archive_name})

            return

        archive_name = self.request.headers['Archive-Name']
        n_frame = self.request.headers['Number-Frame']

        os.system(f'mkdir -p ./dist/recording/{archive_name}/{n_frame}/')

        if t_payload == 'rgb':
            data = np.frombuffer(self.request.body, dtype=np.uint8)
            base = int(np.sqrt(data.shape[0] // 3 // 12))
            width, height = base * 4, base * 3
            data = data.reshape((height, width, 3))

            imageio.imsave(
                f'./dist/recording/{archive_name}/{n_frame}/rgb.png', data)

        elif t_payload == 'depth':
            data = np.frombuffer(self.request.body, dtype=np.float32)
            base = int(np.sqrt(data.shape[0] // 12))
            width, height = base * 4, base * 3
            data = data.reshape((height, width, 1))

            imageio.imsave(
                f'./dist/recording/{archive_name}/{n_frame}/depth.png', data)

            with open(f'./dist/recording/{archive_name}/{n_frame}/depth.bytes', 'wb') as f:
                f.write(self.request.body)

        self.json({'Ok': True})


recording_http_routes = [
    (r"/recording/", RecordingHTTPHandler)
]

__all__ = ['recording_http_routes']
