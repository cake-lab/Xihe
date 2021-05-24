import json
import uuid

import numpy as np
import tornado.websocket

from service.utils import BaseHttpRouter
from service.utils import register_session
from service.utils import register_anchor_size


class SessionHTTPHandler(BaseHttpRouter):
    def post(self):
        sid = uuid.uuid4()

        anchor_size = int(self.request.headers['Anchor-Size'])
        anchors = register_anchor_size(anchor_size)

        register_session(sid, anchors)

        self.json({'ok': True, 'sid': str(sid)})


session_http_routes = [
    (r"/session/", SessionHTTPHandler)
]

__all__ = ['session_http_routes']
