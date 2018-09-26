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
    -   TODO: use 12 labels and ignore the VOID class (i.e., 11 labels)
-   960 x 720 scaled down by factor of 2 to 480 x 360

[CamVid]: http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/
[CamVid-classes]: http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/#ClassLabels

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
| Accuracy                | 0.858503
| mean per class accuracy | 0.596212
| mean I/U                | 0.477470
| Bicyclist               | 0.195386
| Building                | 0.686192
| Car                     | 0.509532
| Column/Pole             | 0.234483
| Fence                   | 0.149444
| Pedestrian              | 0.316118
| Road                    | 0.824710
| Sidewalk                | 0.735486
| SignSymbol              | 0.155891
| Sky                     | 0.869758
| Tree                    | 0.575167

### Qualitative Results

<table>
  <tr>
    <td>
      <img src="https://user-images.githubusercontent.com/2184469/46000452-3ac93300-c06e-11e8-93b3-b8321abdf8a7.png" />
    </td>
    <td>
      <img src="https://user-images.githubusercontent.com/2184469/46000453-3ac93300-c06e-11e8-8d52-13d9bec343e7.png" />
    </td>
    <td>
      <img src="https://user-images.githubusercontent.com/2184469/46000454-3ac93300-c06e-11e8-8141-6d81ca0d1dcb.png" />
    </td>
    <td>
      <img src="https://user-images.githubusercontent.com/2184469/46000455-3ac93300-c06e-11e8-85e0-d43f488e0de4.png" />
    </td>
  </tr>
</table>



## [Bayesian SegNet][Kendall et al. (2015)]

![Bayesian SegNet](https://user-images.githubusercontent.com/2184469/45915765-7bcc0800-be20-11e8-87cf-4d778b1b3837.png)

The following table describes training hyperparameters.

| Crop Size | Epochs | Batch Size | Patience | Optimizer | α    | 𝛃    | α Decay | Dropout | Samples |
|:----------|:-------|:-----------|:---------|:----------|:-----|:-----|:--------|:--------|:--------|
| 352 x 480 | 200    | 8          | 50       | SGD       | 1e-3 | 0.9  | 0.95    | 50%     | 40      |

-   batch normalization statistics computed per batch during training and
    using a rolling average computed over input batches for validation and
    testing
    -   original paper uses a static statistics computed over the training data
-   encoder transfer learning from VGG16 trained on ImageNet
    -   TODO: train the custom version of the VGG16 encoder using dropout for
        better transfer of weights.
-   best model in terms of validation accuracy is kept as final model
-   median frequency balancing of class labels ([Eigen et al. (2014)][])
    -   weighted categorical cross-entropy loss function
-   local contrast normalization of inputs ([LeCun et al. (2009)][])
-   pooling indexes ([Badrinarayanan et al. (2015)][])

### Quantitative Results

The following table outlines the testing results from Bayesian SegNet.

| Metric                  | Test Score |
|:------------------------|:-----------|
| Accuracy                | 0.837571
| mean per class accuracy | 0.602180
| mean I/U                | 0.450497
| Bicyclist               | 0.148188
| Building                | 0.650249
| Car                     | 0.439393
| Column_Pole             | 0.169333
| Fence                   | 0.165258
| Pedestrian              | 0.274567
| Road                    | 0.811342
| Sidewalk                | 0.719864
| SignSymbol              | 0.183497
| Sky                     | 0.854170
| Tree                    | 0.539600

### Qualitative Results

<table>
  <tr>
    <td>
      <img src="https://user-images.githubusercontent.com/2184469/46114992-bb954580-c1ba-11e8-8786-4edfd1284d08.png" />
    </td>
    <td>
      <img src="https://user-images.githubusercontent.com/2184469/46114994-bb954580-c1ba-11e8-805b-5d94b45a615a.png" />
    </td>
    <td>
      <img src="https://user-images.githubusercontent.com/2184469/46114995-bc2ddc00-c1ba-11e8-9621-127b5b6a7122.png" />
    </td>
    <td>
      <img src="https://user-images.githubusercontent.com/2184469/46114996-bc2ddc00-c1ba-11e8-8e74-ee741247102b.png" />
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
| Accuracy                | 0.826657
| mean per class accuracy | 0.659619
| mean I/U                | 0.448913
| Bicyclist               | 0.076790
| Building                | 0.624237
| Car                     | 0.502403
| Column/Pole             | 0.201589
| Fence                   | 0.115558
| Pedestrian              | 0.254075
| Road                    | 0.828246
| Sidewalk                | 0.745941
| SignSymbol              | 0.153115
| Sky                     | 0.879440
| Tree                    | 0.556651

### Qualitative Results

<table>
  <tr>
    <td>
      <img src="https://user-images.githubusercontent.com/2184469/46000488-52082080-c06e-11e8-9787-d35d1dec990a.png" />
    </td>
    <td>
      <img src="https://user-images.githubusercontent.com/2184469/46000489-52a0b700-c06e-11e8-8d5b-2f33aa1995a6.png" />
    </td>
    <td>
      <img src="https://user-images.githubusercontent.com/2184469/46000490-52a0b700-c06e-11e8-89dc-45e93cd6cbcf.png" />
    </td>
    <td>
      <img src="https://user-images.githubusercontent.com/2184469/46000492-52a0b700-c06e-11e8-937e-95d4cb53b3ff.png" />
    </td>
  </tr>
</table>



## [Bayesian Tiramisu][Kendall et al. (2017)]

### Quantitative Results

The following table outlines the testing results from Bayesian Tiramisu. Only
the hybrid model (aleatoric + epistemic uncertainty) is shown for brevity.

### Qualitative Results


<!-- References -->

[LeCun et al. (2009)]: http://yann.lecun.com/exdb/publis/pdf/jarrett-iccv-09.pdf
[Eigen et al. (2014)]: https://arxiv.org/abs/1411.4734
[Badrinarayanan et al. (2015)]: https://arxiv.org/pdf/1511.00561.pdf
[Kendall et al. (2015)]: https://arxiv.org/abs/1511.02680
[Jégou et al. (2016)]: https://arxiv.org/abs/1611.09326
[Kendall et al. (2017)]: http://papers.nips.cc/paper/7141-what-uncertainties-do-we-need-in-bayesian-deep-learning-for-computer-vision
