from .dump import dump_http_routes
from .session import session_http_routes
from .recording import recording_http_routes
from .network_testing import network_testing_http_routes
from .lighting_estimation import lighting_estimation_http_routes


r = [
    *dump_http_routes,
    *session_http_routes,
    *recording_http_routes,
    *network_testing_http_routes,
    *lighting_estimation_http_routes
]
api_v2_http_routes = [(f'/api/v2{v[0]}', v[1]) for v in r]

__all__ = ['api_v2_http_routes']
