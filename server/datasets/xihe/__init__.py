import importlib
from typing import Union


def gen_data(dataset: str = 'test', index: Union[str, int] = 'all'):
    """Generate Xihe dataset

    Parameters
    ----------
    dataset : str
        Dataset to generate: train, traind10, test
    index : str | int
        The index of data item to generate, 'all' for all
    """

    # Use importlib to import modules for delaying CUDA initialization
    print("Generating dataset")
    importlib.import_module(
        'datasets.xihe.preprocess.pack'
    ).generate(dataset, index=index)

    print("Packing dataset")
    importlib.import_module(
        'datasets.xihe.preprocess.pack'
    ).pack(dataset, index=index)


def gen_fps_data(dataset: str = 'test', index: Union[str, int] = 'all', n_points: int = 1280):
    print("Generating dataset")
    importlib.import_module(
        'datasets.xihe.preprocess.sample_fps'
    ).sample_fps(dataset, index=index, n_points=n_points)
