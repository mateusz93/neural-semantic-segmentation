# The 100 Layers Tiramisu (Implementation)

[![build-status][]][ci-server]

[build-status]: https://travis-ci.com/Kautenja/the-100-layers-tiramisu.svg?branch=master
[ci-server]: https://travis-ci.com/Kautenja/the-100-layers-tiramisu

An Implementation of
[The One Hundred Layers Tiramisu](https://arxiv.org/abs/1611.09326) using
[Keras](https://keras.io/).

## Installation

To install requirements for the project:

```shell
python -m pip install -r requirements.txt
```

## Validation Results

Results were generated using a machine equipped with  128GB RAM, nVidia P100
GPU, and Intel Xeon CPU @ 2.10GHz.

### 11 Class

[Train-Tiramisu103-CamVid11.ipynb](Train-Tiramisu103-CamVid11.ipynb) generates
these results for the 11 class version of CamVid defined in the mapping file
[11_class.txt](11_class.txt).

<!-- TODO: images -->

#### Metrics

<!-- TODO: metrics table -->

### 32 Class

[Train-Tiramisu103-CamVid32.ipynb](Train-Tiramisu103-CamVid32.ipynb) generates
these results for the full 32 class version of CamVid.

<!-- TODO: images -->

#### Metrics

<!-- TODO: metrics table -->
