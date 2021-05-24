# Xihe Server Code

## Installation

This repo is developed with Python, CUDA C++, and bash scripts. Please use the following commands to install the python dependencies first.

```bash
pipenv shell
pipenv install

pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html --upgrade
```

## Dataset and Model

One of the key steps in reproducing our work is to generate the transformed point cloud datasets. For simplicity, we provide a pre-generated testing dataset at the `data/dataset/` directory.

Please unzip the `data/dataset/pointar.zip`, `data/dataset/xihe.zip` and `data/dataset/xihe_fps.zip` files, and change the corresponding path definition in `config.py`.

Our pre-trained XiheNet model can be found at `server/model/model.ckpt`. To train the XiheNet from scratch, you will need to have access to the training dataset. Due to the large size of the training dataset (~TBs), it is difficult to provide directly---instead we provide the [instructions](#generating-training-dataset) for you to generate the dataset below.

## Start the Service

Use the following code to start the Xihe server:

```bash
./launch.py serve
```

## Directory Structure

- `datasets`: datasets definitions and loaders.
- `model`: XiheNet model definition and checkpoints
- `service`: Xihe backend service
- `utils3d`: essential 3D transformation and utilities for 3D math operations.
- `configs.py`: path definition for datasets.
- `launch.py`: command line entry script
- `Pipfile`: pipenv dependency definition file

## Generating Training Dataset

(The instructions below are adapted from the one provided in [PointAR](https://github.com/cake-lab/PointAR).)

We provide the scripts to generate their respective datasets. At the high level,

- users should first obtain access to the two open-source datasets (i.e., [Matterport3D]( https://github.com/niessner/Matterport) and [Neural Illumination](https://illumination.cs.princeton.edu) datasets);
- download these two datasets to a desirable directory. For the Matterport3D dataset, unzip the downloaded zip files and place them in a directory with structure similar to `v1/scans/<SCENE_ID>/...`. For the Neural Illumination dataset, just store the downloaded zip files, i.e. `illummaps_<SCENE_ID>.zip`, directly in a directory.
- modify the corresponding the path variable in `config.py` file to reflect the local directory name;
- then run the following commands to start generation.
  - PointAR: `./launch.py gen_pointar_data --dataset=test/train`
  - Xihe: `./launch.py gen_xihe_data --dataset=test/train`
  - XiheFPS: `./launch.py gen_xihe_fps_data --dataset=test/train`

Note it can take a few hours to generate the entire dataset (~1.4TB) depending on the GPU devices.
