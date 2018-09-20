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

### SegNet

-   [x] median frequency balancing of class labels [Eigen et al. (2014)](https://arxiv.org/abs/1411.4734)
-   [x] local contrast normalization of inputs [LeCun et al. (2009)](http://yann.lecun.com/exdb/publis/pdf/jarrett-iccv-09.pdf)
-   [x] pooling indexes [Badrinarayanan et al. (2016)](https://arxiv.org/pdf/1511.00561.pdf)

The following table outlines the testing results from SegNet.

| Metric          | Validation |
|:----------------|:-----------|
| Accuracy        | 0.840661
| mean IoU        | 0.485506
| Bicyclist       | 0.149914
| Building        | 0.688313
| Car             | 0.639180
| Column_Pole     | 0.207784
| Fence           | 0.162744
| Pedestrian      | 0.262279
| Road            | 0.828501
| Sidewalk        | 0.696738
| SignSymbol      | 0.161242
| Sky             | 0.896308
| Tree            | 0.647563

<table style="width:100%">
  <tr>
    <td>
      <img src="https://user-images.githubusercontent.com/2184469/45790933-f5ab9800-bc4c-11e8-92ec-d867022647a5.png" />
    </td>
    <td>
      <img src="https://user-images.githubusercontent.com/2184469/45790934-f5ab9800-bc4c-11e8-9cf3-bd4d1a752a65.png" />
    </td>
    <td>
      <img src="https://user-images.githubusercontent.com/2184469/45790935-f5ab9800-bc4c-11e8-82d2-ce8f80e9c706.png" />
    </td>
  </tr>
</table>

<!-- ### Tiramisu

-   [x] median frequency balancing of class labels [Eigen et al. (2014)](https://arxiv.org/abs/1411.4734)
-   [x] local contrast normalization of inputs [LeCun et al. (2009)](http://yann.lecun.com/exdb/publis/pdf/jarrett-iccv-09.pdf)

The following table outlines the testing results from Tiramisu (103 layer).

| Metric          | Validation |
|:----------------|:-----------|
| Accuracy        | 0.840661
| mean IoU        | 0.485506
| Bicyclist       | 0.149914
| Building        | 0.688313
| Car             | 0.639180
| Column_Pole     | 0.207784
| Fence           | 0.162744
| Pedestrian      | 0.262279
| Road            | 0.828501
| Sidewalk        | 0.696738
| SignSymbol      | 0.161242
| Sky             | 0.896308
| Tree            | 0.647563

<table style="width:100%">
  <tr>
    <td>
      <img src="https://user-images.githubusercontent.com/2184469/45790933-f5ab9800-bc4c-11e8-92ec-d867022647a5.png" />
    </td>
    <td>
      <img src="https://user-images.githubusercontent.com/2184469/45790934-f5ab9800-bc4c-11e8-9cf3-bd4d1a752a65.png" />
    </td>
    <td>
      <img src="https://user-images.githubusercontent.com/2184469/45790935-f5ab9800-bc4c-11e8-82d2-ce8f80e9c706.png" />
    </td>
  </tr>
</table> -->
