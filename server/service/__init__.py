import tornado
from tornado.log import enable_pretty_logging
from service.api import api_v2_http_routes


enable_pretty_logging()


def start_service(port=8550, debug=True):
    """ Holds all the registered HTTP endpoints

    input: All the endpoints should be defined under the routes directory

    calling this function will also setup the gRPC connection between
    front-end web server and the Triton inference server
    """

    app = tornado.web.Application(
        [*api_v2_http_routes],
        debug=debug,
        autoreload=debug)

    app.listen(port)
    print('Tornado Server Started...')
    tornado.ioloop.IOLoop.current().start()
