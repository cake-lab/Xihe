import uuid
import json
import tornado.web
import numpy as np
from urllib.parse import unquote

from utils3d import fibonacci_sphere


class BaseHttpRouter(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header('Content-Type', 'application/json')

    def get_body_json(self):
        body = self.request.body.decode('utf-8')
        body = json.loads(unquote(body))
        return body

    def json(self, data):
        self.write(json.dumps(data))


anchor_pool = {}

anchor_pool[2048] = fibonacci_sphere(2048)
anchor_pool[1280] = fibonacci_sphere(1280)
anchor_pool[1024] = fibonacci_sphere(1024)
anchor_pool[768] = fibonacci_sphere(768)
anchor_pool[512] = fibonacci_sphere(512)


session_pool = {}


def register_session(sid: uuid.UUID, anchors: np.ndarray):
    if sid not in session_pool:
        session_pool[sid] = {
            'anchors': anchors,
            'point_clouds': []
        }
    else:
        print('Error, SID conflict')


def register_anchor_size(samples: int):
    if samples not in anchor_pool:
        anchor_pool[samples] = fibonacci_sphere(samples)

    return anchor_pool[samples]
