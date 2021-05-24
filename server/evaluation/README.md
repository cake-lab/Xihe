# Evaluation

## Paper Results
This directory includes the code and data for reproducing all the results presented in the paper. The experiment results are presented as jupyter notebooks.

- `uspc`: Figure 2, 3, 10, 11, 12
- `profile_client`: Table 2, Table 4
- `profile_end2end`: Figure 7
- `profile_networking`: Figure 9, 13
- `training`: Figure 14 (a)
- `profile_serving`: Figure 14 (b)
- `trigger`: Figure 15
- `real_world_testing`: Table 5


## Server-side Experiments
We also include scripts and data for reproducing all server-side experiments.

- `training`: you will need to have access to a pre-trained XiheNet and three test datasets (totalling ~9GB). For reference, we provide a trained model at [here]() and test datasets for downloading at [here](). Alternatively, you could follow the dataset generation steps described in [readme.md](../readme.md) to generate all three training/testing datasets.
- `profile_serving`: note, you will need to first install PyTorch and torch_cluster by following the [readme.md](../readme.md)


## Client-side Experiments
To reproduce client-side experiments (without environmental sensing), the easiest way is to follow the [Xihe-client instruction]() to build a Unity app for the target platform (e.g., OSX). Once done, you should be able to reproduce all remaining experiments with our AR session recorder.
- If you have a Lidar-enabled iPad or iPhone, you can test the AR session recorder.
