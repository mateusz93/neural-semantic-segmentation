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

## [CamVid][]

-   [32 classes][32-class] generalized to 11 classes using mapping in [11_class.txt](11_class.txt)
-   960 x 720 scaled down by factor of 2 to 480 x 360

[CamVid]: http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/
[32-class]: http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/#ClassLabels

## [SegNet][Badrinarayanan et al. (2015)]

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

The following table describes training hyperparameters.

| Crop Size | Epochs | Batch Size | Patience | Optimizer | α    | 𝛃    | α Decay |
|:----------|:-------|:-----------|:---------|:----------|:-----|:-----|:--------|
| 352 x 480 | 1000   | 8          | 50       | SGD       | 1e-3 | 0.9  | 0.95    |

-   batch normalization statistics computed per batch during training and
    using a rolling average for validation and testing
    -   TODO: change to calculate static mean and variance over training data
        (i.e., calculate the rolling averages over 1 epoch of training data,
        then freeze the values for validation and testing)
-   random vertical flips of images for training, validation, and testing
    -   TODO: disable this to follow paper
-   encoder transfer learning from VGG16 trained on ImageNet
-   best model in terms of training loss is kept as final model
    -   TODO: change to keep the best in terms of validation
-   median frequency balancing of class labels ([Eigen et al. (2014)][])
    -   weighted categorical cross-entropy loss function
-   local contrast normalization of inputs ([LeCun et al. (2009)][])
-   pooling indexes ([Badrinarayanan et al. (2015)][])

### Quantitative Results

The following table outlines the testing results from SegNet.

| Metric      | Test Score |
|:------------|:-----------|
| Accuracy    | 0.846443
| mean I/U    | 0.447931
| Bicyclist   | 0.093297
| Building    | 0.658801
| Car         | 0.473069
| Column_Pole | 0.218206
| Fence       | 0.131719
| Pedestrian  | 0.230519
| Road        | 0.825075
| Sidewalk    | 0.715983
| SignSymbol  | 0.149735
| Sky         | 0.871027
| Tree        | 0.559813

### Qualitative Results

<table>
  <tr>
    <td>
      <img src="https://user-images.githubusercontent.com/2184469/45915621-8c2eb380-be1d-11e8-825e-764e9ebab4c5.png" />
    </td>
    <td>
      <img src="https://user-images.githubusercontent.com/2184469/45915622-8cc74a00-be1d-11e8-9366-0c4670fcc678.png" />
    </td>
    <td>
      <img src="https://user-images.githubusercontent.com/2184469/45915623-8cc74a00-be1d-11e8-9fc0-e2d82e331a41.png" />
    </td>
    <td>
      <img src="https://user-images.githubusercontent.com/2184469/45915624-8cc74a00-be1d-11e8-917b-44ef09082f6f.png" />
    </td>
  </tr>
</table>



## [Bayesian SegNet][Kendall et al. (2015)]

![Bayesian SegNet](https://user-images.githubusercontent.com/2184469/45915765-7bcc0800-be20-11e8-87cf-4d778b1b3837.png)

The following table describes training hyperparameters.

| Crop Size | Epochs | Batch Size | Patience | Optimizer | α    | 𝛃    |
|:----------|:-------|:-----------|:---------|:----------|:-----|:-----|
| 352 x 480 | 1000   | 8          | 50       | SGD       | 1e-3 | 0.9  |

-   batch normalization statistics computed per batch during training and
    using a rolling average for validation and testing
    -   TODO: change to calculate static mean and variance over training data
        (i.e., calculate the rolling averages over 1 epoch of training data,
        then freeze the values for validation and testing)
-   random vertical flips of images for training, validation, and testing
    -   TODO: disable this to follow paper
-   encoder transfer learning from VGG16 trained on ImageNet
    -   TODO: does this make sense given that VGG16 was _not_ trained with
        dropout? they don't mention transfer learning in the paper. probably
        best to disable this functionality
-   best model in terms of training loss is kept as final model
    -   TODO: change to keep the best in terms of validation
-   median frequency balancing of class labels ([Eigen et al. (2014)][])
    -   weighted categorical cross-entropy loss function
-   local contrast normalization of inputs ([LeCun et al. (2009)][])
-   pooling indexes ([Badrinarayanan et al. (2015)][])
-   dropout rate of 50% during training and inference
-   50 Monte Carlo samples to estimate mean class and variance

### Quantitative Results

The following table outlines the testing results from SegNet.

| Metric      | Test Score |
|:------------|:-----------|
| Accuracy    | 0.785175
| mean I/U    | 0.389309
| Bicyclist   | 0.069309
| Building    | 0.552303
| Car         | 0.399653
| Column_Pole | 0.117000
| Fence       | 0.079420
| Pedestrian  | 0.199412
| Road        | 0.779842
| Sidewalk    | 0.644177
| SignSymbol  | 0.103987
| Sky         | 0.822849
| Tree        | 0.514446

### Qualitative Results

<table>
  <tr>
    <td>
      <img src="https://user-images.githubusercontent.com/2184469/45964351-2c860300-bfeb-11e8-83a8-2aae9b8ffc61.png" />
    </td>
    <td>
      <img src="https://user-images.githubusercontent.com/2184469/45964349-2c860300-bfeb-11e8-8451-9c71c8249b33.png" />
    </td>
    <td>
      <img src="https://user-images.githubusercontent.com/2184469/45964348-2c860300-bfeb-11e8-9dd2-05c3d3ac2552.png" />
    </td>
    <td>
      <img src="https://user-images.githubusercontent.com/2184469/45964347-2c860300-bfeb-11e8-9d95-a7eb7b72fb3b.png" />
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

The following table describes training hyperparameters.

| Crop Size | Epochs | Batch Size | Patience | Optimizer | α    | α Decay |
|:----------|:-------|:-----------|:---------|:----------|:-----|:--------|
| 224 x 224 | 200    | 3          | 100      | RMSprop   | 1e-3 | 0.995   |
| 352 x 480 | 200    | 1          | 50       | RMSprop   | 1e-4 | 1.000   |

-   random vertical flips of images for training, validation, and testing
    -   TODO: disable this for validation and testing
-   batch normalization statistics computed _per batch_ during training and
    inference
-   median frequency balancing of class labels ([Eigen et al. (2014)][])
    -   weighted categorical cross-entropy loss function
-   local contrast normalization of inputs ([LeCun et al. (2009)][])
-   skip connections ([Jégou et al. (2016)][])

### Quantitative Results

The following table outlines the testing results from 103 Layers Tiramisu.

| Metric                  | Test Score |
|:------------------------|:-----------|
| Accuracy                | 0.814069
| mean per class Accuracy | 0.648450
| mean I/U                | 0.443015
| Bicyclist               | 0.080975
| Building                | 0.627993
| Car                     | 0.447337
| Column_Pole             | 0.211398
| Fence                   | 0.221945
| Pedestrian              | 0.244067
| Road                    | 0.806656
| Sidewalk                | 0.705870
| SignSymbol              | 0.126961
| Sky                     | 0.878069
| Tree                    | 0.521894

### Qualitative Results

<table>
  <tr>
    <td>
      <img src="https://user-images.githubusercontent.com/2184469/45962925-a2886b00-bfe7-11e8-8fc7-13c94a789568.png" />
    </td>
    <td>
      <img src="https://user-images.githubusercontent.com/2184469/45962924-a1efd480-bfe7-11e8-9fd4-9c62d39e9e7b.png" />
    </td>
    <td>
      <img src="https://user-images.githubusercontent.com/2184469/45962923-a1efd480-bfe7-11e8-96c0-a473a6c19383.png" />
    </td>
    <td>
      <img src="https://user-images.githubusercontent.com/2184469/45962922-a1efd480-bfe7-11e8-8a8d-cfb85928fede.png" />
    </td>
  </tr>
</table>



## [Bayesian Tiramisu][Kendall et al. (2017)]

### Quantitative Results

The following table outlines the testing results from SegNet.

### Qualitative Results


<!-- References -->

[LeCun et al. (2009)]: http://yann.lecun.com/exdb/publis/pdf/jarrett-iccv-09.pdf
[Eigen et al. (2014)]: https://arxiv.org/abs/1411.4734
[Badrinarayanan et al. (2015)]: https://arxiv.org/pdf/1511.00561.pdf
[Kendall et al. (2015)]: https://arxiv.org/abs/1511.02680
[Jégou et al. (2016)]: https://arxiv.org/abs/1611.09326
[Kendall et al. (2017)]: http://papers.nips.cc/paper/7141-what-uncertainties-do-we-need-in-bayesian-deep-learning-for-computer-vision
