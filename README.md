# Neural Semantic Segmentation

Implementations of neural network papers for semantic segmentation using Keras
and TensorFlow.

## Installation

To install requirements for the project:

```shell
python -m pip install -r requirements.txt
```

## Hardware Specification

Results were generated using a machine equipped with  128GB RAM, nVidia P100
GPU, and Intel Xeon CPU @ 2.10GHz. All results shown are from the testing
dataset.

## [SegNet][Badrinarayanan et al. (2016)]

<table>
  <tr>
    <td>
      <img alt="SegNet" src="https://user-images.githubusercontent.com/2184469/45845186-1118b080-bcea-11e8-967f-d1d0b9d93bb8.png" />
    </td>
    <td>
      <img alt="Pooling Indexes" src="https://user-images.githubusercontent.com/2184469/45845185-1118b080-bcea-11e8-8fb3-82ebb3f15ea6.png" />
    </td>
  </tr>
</table>

-   [x] encoder transfer learning from VGG16 trained on ImageNet
-   [x] optimized using SGD with 𝛃=0.9, α=0.001 (constant)
-   [x] trained for 50 epochs with a batch size of 4 ([Badrinarayanan et al. (2016)][] used 12)
-   [x] best model in terms of training loss is kept as final model
-   [x] median frequency balancing of class labels ([Eigen et al. (2014)][])
    -   weighted categorical cross-entropy loss function
-   [x] local contrast normalization of inputs ([LeCun et al. (2009)][])
-   [x] pooling indexes ([Badrinarayanan et al. (2016)][])

### Quantitative Results

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

### Qualitative Results

<table>
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

## [The One Hundred Layers Tiramisu][Jégou et al. (2016)]

<table>
  <tr>
    <td>
        <img alt="103 Layers Tiramisu" src="https://user-images.githubusercontent.com/2184469/45852685-a88bfc80-bd06-11e8-9ea1-9044144b1442.png">
    </td>
    <td>
        <img alt="Dense Blocks" src="https://user-images.githubusercontent.com/2184469/45852691-aa55c000-bd06-11e8-865b-b852485b40af.png">
    </td>
  </tr>
</table>

-   [x] trained on 224 x 224 for for 50 epochs with batch size 4 (patience 100)
    -   optimized using RMSprop with learning rate 0.001, decay 0.995
-   [x] trained on 352 (360) x 480 for for 50 epochs with batch size 1 (patience 50)
    -   optimized using RMSprop with learning rate 0.001, decay 0.995
-   [x] median frequency balancing of class labels ([Eigen et al. (2014)][])
    -   weighted categorical cross-entropy loss function
-   [x] local contrast normalization of inputs ([LeCun et al. (2009)][])
-   [x] skip connections ([Jégou et al. (2016)][])

### Quantitative Results

The following table outlines the testing results from 103 Layers Tiramisu.



<!-- References -->

[LeCun et al. (2009)]: http://yann.lecun.com/exdb/publis/pdf/jarrett-iccv-09.pdf
[Eigen et al. (2014)]: https://arxiv.org/abs/1411.4734
[Badrinarayanan et al. (2016)]: https://arxiv.org/pdf/1511.00561.pdf
[Jégou et al. (2016)]: https://arxiv.org/abs/1611.09326
