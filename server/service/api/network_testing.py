import glob
import base64
import numpy as np

from service.utils import BaseHttpRouter


class NetworkTestingDataDecoder:
    def naive_decode(self, data):
        anchor_clr = base64.decodebytes(data.encode())
        anchor_clr = np.frombuffer(anchor_clr, dtype=np.float32)

        return anchor_clr

    def naive_bytes_decode(self, data):
        anchor_clr = np.frombuffer(data, dtype=np.float32)
        return anchor_clr

    def xihe_decode(self, data):
        data = data.split(',')

        bytes_clr = base64.decodebytes(data[1].encode())
        bytes_idx = base64.decodebytes(data[0].encode())

        arr_idx = np.frombuffer(bytes_idx, dtype=np.uint16)
        arr_clr = np.frombuffer(bytes_clr, dtype=np.float32)

        arr_clr = arr_clr.reshape((-1, 3))

        anchor_clr = np.zeros((1280, 3), dtype=np.float32)
        anchor_clr[arr_idx] = arr_clr

        return anchor_clr

    def xihe_bytes_decode(self, data):
        pivot = int.from_bytes(data[:2], byteorder='little')

        bytes_idx = data[2:pivot * 2 + 2]
        bytes_clr = data[pivot * 2 + 2:]

        arr_idx = np.frombuffer(bytes_idx, dtype=np.uint16)
        arr_clr = np.frombuffer(bytes_clr, dtype=np.float32)

        arr_clr = arr_clr.reshape((-1, 3))

        anchor_clr = np.zeros((1280, 3), dtype=np.float32)
        anchor_clr[arr_idx] = arr_clr

        return anchor_clr

    def xihe_bytes_decode_fast(self, data):
        arr_bytes = np.frombuffer(data, dtype=np.byte)
        arr_bytes = arr_bytes.reshape((-1, (2 + 4 * 3)))

        arr_idx = np.frombuffer(arr_bytes[:, :2].tobytes(), dtype=np.uint16)
        arr_clr = np.frombuffer(arr_bytes[:, 2:].tobytes(), dtype=np.float32)
        arr_clr = arr_clr.reshape((-1, 3))

        anchor_clr = np.zeros((1280, 3), dtype=np.float32)
        anchor_clr[arr_idx] = arr_clr

        return anchor_clr


class NetworkTestingLogHTTPRouter(BaseHttpRouter, NetworkTestingDataDecoder):
    def post(self):
        # body = self.get_body_json()

        if self.request.headers['encoding'] == 'xihe':
            anchor_clr = self.xihe_bytes_decode_fast(self.request.body)
        elif self.request.headers['encoding'] == 'naive':
            anchor_clr = self.naive_bytes_decode(self.request.body)
        else:
            self.set_status(400)
            return

        self.json({'ok': True})


class NetworkTestingClientLogFileHTTPRouter(BaseHttpRouter):
    def post(self):
        body = self.get_body_json()
        client_log = body['data']

        g = glob.glob('./dump/network-testing/*_client.csv')
        f = open(f'./dump/network-testing/{len(g)}_client.csv', 'w')
        f.write(client_log)
        f.close()

        print('New measurement results received')

        self.json({'ok': True})


network_testing_http_routes = [
    (r"/network-testing/log/", NetworkTestingLogHTTPRouter),
    (r"/network-testing/client-log/", NetworkTestingClientLogFileHTTPRouter)
]

__all__ = ['network_testing_http_routes']
