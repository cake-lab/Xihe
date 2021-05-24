import importlib
from typing import Union


def gen_data(dataset: str = 'test', index: Union[str, int] = 'all'):
    """Generate PointAR dataset

    Parameters
    ----------
    dataset : str
        Dataset to generate: train, traind10, test
    index : str | int
        The index of data item to generate, 'all' for all
    """

    # Use importlib to import modules for delaying CUDA initialization
    print("Generating test dataset")
    importlib.import_module(
        'datasets.pointar.preprocess.pack'
    ).generate(dataset, index=index)

    print("Packing dataset")
    importlib.import_module(
        'datasets.pointar.preprocess.pack'
    ).pack(dataset, index=index)
