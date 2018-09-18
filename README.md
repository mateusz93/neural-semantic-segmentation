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

The following results use the reduced 11 class version of the dataset defined
by the mapping file [11_class.txt](11_class.txt).
[Train-Tiramisu103-CamVid11.ipynb](Train-Tiramisu103-CamVid11.ipynb) generates
these results.

<!-- TODO: images -->

#### Metrics

<!-- TODO: metrics table -->

### 32 Class

The following results use the dataset with all 32 original class labels.
[Train-Tiramisu103-CamVid32.ipynb](Train-Tiramisu103-CamVid32.ipynb) generates
these results.

<!-- TODO: images -->

#### Metrics

<!-- TODO: metrics table -->
