# Directly reading from file system is faster
# but it requires more disk spaces than hdf5 packaged data
from datasets.pointar.loader.fs import PointARTestDataset
from datasets.pointar.loader.fs import PointARTrainDataset
from datasets.pointar.loader.fs import PointARTrainD10Dataset
