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

-   [32 classes][CamVid-classes] generalized to 11 classes using mapping in
    [11_class.txt](11_class.txt)
    -   use 12 labels and ignore the Void class (i.e., 11 labels)
-   960 x 720 scaled down by factor of 2 to 480 x 360

[CamVid]: http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/
[CamVid-classes]: http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/#ClassLabels

## [SegNet][Badrinarayanan et al. (2015)]

<table>
  <tr>
    <td>
        <img alt="SegNet" src="img/segnet/model.png">
    </td>
    <td>
        <img alt="Pooling Indexes" src="img/segnet/max-pooling.png">
    </td>
  </tr>
</table>

The following table describes training hyperparameters.

| Crop Size | Epochs | Batch Size | Patience | Optimizer | α    | 𝛃    | α Decay |
|:----------|:-------|:-----------|:---------|:----------|:-----|:-----|:--------|
| 352 x 480 | 200    | 8          | 50       | SGD       | 1e-3 | 0.9  | 0.95    |

-   batch normalization statistics computed per batch during training and
    using a rolling average computed over input batches for validation and
    testing
    -   original paper uses a static statistics computed over the training data
-   encoder transfer learning from VGG16 trained on ImageNet
-   best model in terms of validation accuracy is kept as final model
-   median frequency balancing of class labels ([Eigen et al. (2014)][])
    -   weighted categorical cross-entropy loss function
-   local contrast normalization of inputs ([LeCun et al. (2009)][])
-   pooling indexes ([Badrinarayanan et al. (2015)][])

### Quantitative Results

The following table outlines the testing results from SegNet.

| Metric                  | Test Score |
|:------------------------|:-----------|
| Global Accuracy         | 0.888466
| Mean Per Class Accuracy | 0.593288
| Mean I/U                | 0.495996
| Bicyclist               | 0.301233
| Building                | 0.693133
| Car                     | 0.503114
| Column/Pole             | 0.218290
| Fence                   | 0.152144
| Pedestrian              | 0.332084
| Road                    | 0.887113
| Sidewalk                | 0.756714
| Sign                    | 0.161815
| Sky                     | 0.871147
| Vegetation              | 0.579169

### Qualitative Results

<table>
  <tr>
    <td>
      <img src="img/segnet/0.png" />
    </td>
    <td>
      <img src="img/segnet/1.png" />
    </td>
    <td>
      <img src="img/segnet/2.png" />
    </td>
    <td>
      <img src="img/segnet/3.png" />
    </td>
  </tr>
</table>



## [Bayesian SegNet][Kendall et al. (2015)]

![Bayesian SegNet](img/bayesian-segnet/model.png)

The following table describes training hyperparameters.

| Crop Size | Epochs | Batch Size | Patience | Optimizer | α    | 𝛃    | α Decay | Dropout | Samples |
|:----------|:-------|:-----------|:---------|:----------|:-----|:-----|:--------|:--------|:--------|
| 352 x 480 | 200    | 8          | 50       | SGD       | 1e-3 | 0.9  | 0.95    | 50%     | 40      |

-   batch normalization statistics computed per batch during training and
    using a rolling average computed over input batches for validation and
    testing
    -   original paper uses a static statistics computed over the training data
-   encoder transfer learning from VGG16 trained on ImageNet
    -   note that VGG16 does not have any dropout by default; transfer from a
        Bayesian VGG16 model could improve results
-   best model in terms of validation accuracy is kept as final model
-   median frequency balancing of class labels ([Eigen et al. (2014)][])
    -   weighted categorical cross-entropy loss function
-   local contrast normalization of inputs ([LeCun et al. (2009)][])
-   pooling indexes ([Badrinarayanan et al. (2015)][])

### Quantitative Results

The following table outlines the testing results from Bayesian SegNet.

| Metric                  | Test Score |
|:------------------------|:-----------|
| Global Accuracy         | 0.854635
| Mean Per Class Accuracy | 0.634437
| Mean I/U                | 0.444531
| Bicyclist               | 0.097716
| Building                | 0.624091
| Car                     | 0.470470
| Column/Pole             | 0.180791
| Fence                   | 0.117226
| Pedestrian              | 0.252237
| Road                    | 0.866110
| Sidewalk                | 0.712532
| Sign                    | 0.148314
| Sky                     | 0.860612
| Vegetation              | 0.559742

### Qualitative Results

<table>
  <tr>
    <td>
      <img src="img/bayesian-segnet/0.png" />
    </td>
    <td>
      <img src="img/bayesian-segnet/1.png" />
    </td>
    <td>
      <img src="img/bayesian-segnet/2.png" />
    </td>
    <td>
      <img src="img/bayesian-segnet/3.png" />
    </td>
  </tr>
</table>



## [The One Hundred Layers Tiramisu][Jégou et al. (2016)]

<table>
  <tr>
    <td>
        <img alt="103 Layers Tiramisu" src="img/tiramisu/model.png">
    </td>
    <td>
        <img alt="Layers" src="img/tiramisu/layers.png">
    </td>
  </tr>
</table>

The following table describes training hyperparameters.

| Crop Size | Epochs | Batch Size | Patience | Optimizer | α    | α Decay | Dropout |
|:----------|:-------|:-----------|:---------|:----------|:-----|:--------|:--------|
| 224 x 224 | 200    | 3          | 100      | RMSprop   | 1e-3 | 0.995   | 20%     |
| 352 x 480 | 200    | 1          | 50       | RMSprop   | 1e-4 | 1.000   | 20%     |

-   random vertical flips of images during training
-   batch normalization statistics computed _per batch_ during training,
    validation, and testing
-   median frequency balancing of class labels ([Eigen et al. (2014)][])
    -   weighted categorical cross-entropy loss function
-   local contrast normalization of inputs ([LeCun et al. (2009)][])
-   skip connections ([Jégou et al. (2016)][])

### Quantitative Results

The following table outlines the testing results from 103 Layers Tiramisu.

| Metric                  | Test Score |
|:------------------------|:-----------|
| Global Accuracy         | 0.835221
| Mean Per Class Accuracy | 0.652682
| Mean I/U                | 0.443558
| Bicyclist               | 0.098166
| Building                | 0.596728
| Car                     | 0.448302
| Column/Pole             | 0.197154
| Fence                   | 0.141940
| Pedestrian              | 0.209208
| Road                    | 0.879248
| Sidewalk                | 0.723948
| Sign                    | 0.138548
| Sky                     | 0.892510
| Vegetation              | 0.553382

### Qualitative Results

<table>
  <tr>
    <td>
      <img src="img/tiramisu/0.png" />
    </td>
    <td>
      <img src="img/tiramisu/1.png" />
    </td>
    <td>
      <img src="img/tiramisu/2.png" />
    </td>
    <td>
      <img src="img/tiramisu/3.png" />
    </td>
  </tr>
</table>


<!--
## [Bayesian Tiramisu][Kendall et al. (2017)]

### Quantitative Results

The following table outlines the testing results from Bayesian Tiramisu. Only
the hybrid model (aleatoric + epistemic uncertainty) is shown for brevity.

### Qualitative Results
-->

<!-- References -->

[LeCun et al. (2009)]: http://yann.lecun.com/exdb/publis/pdf/jarrett-iccv-09.pdf
[Eigen et al. (2014)]: https://arxiv.org/abs/1411.4734
[Badrinarayanan et al. (2015)]: https://arxiv.org/pdf/1511.00561.pdf
[Kendall et al. (2015)]: https://arxiv.org/abs/1511.02680
[Jégou et al. (2016)]: https://arxiv.org/abs/1611.09326
[Kendall et al. (2017)]: http://papers.nips.cc/paper/7141-what-uncertainties-do-we-need-in-bayesian-deep-learning-for-computer-vision
