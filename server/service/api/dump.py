import numpy as np

from service.utils import BaseHttpRouter
from service.utils import anchor_pool
from service.payload import payload_processors


class DumpHTTPHandler(BaseHttpRouter):
    dump_root: str = './dist/xihe_service'

    def post(self):
        f_type = self.request.headers['File-Type']
        f_name = self.request.headers['File-Name']
        anchor_size = self.request.headers['Anchor-Size'] \
            if 'Anchor-Size' in self.request.headers else None

        processor = payload_processors[f_type]

        if 'point_cloud' in f_type:
            if anchor_size is None:
                pc = processor(self.request.body)
            else:
                pc = processor(self.request.body,
                               anchor_pool[int(anchor_size)])
            np.save(f'{self.dump_root}/{f_name}', pc)

        elif 'log' in f_type:
            processor(self.request.body)

        elif 'ar-session' in f_type:
            processor(self.request.body)

        self.json({'OK': True})


dump_http_routes = [
    (r"/dump/", DumpHTTPHandler)
]

__all__ = ['dump_http_routes']
