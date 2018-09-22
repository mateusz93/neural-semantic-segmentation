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

-   encoder transfer learning from VGG16 trained on ImageNet
-   optimized using SGD with 𝛃=0.9, initial α=0.001
    -   learning rate decay of 0.95 per epoch
-   trained for for max of 200 epochs with batch size 8
    -   training stopped early based on training loss with patience 10
-   best model in terms of training loss is kept as final model
-   median frequency balancing of class labels ([Eigen et al. (2014)][])
    -   weighted categorical cross-entropy loss function
-   local contrast normalization of inputs ([LeCun et al. (2009)][])
-   pooling indexes ([Badrinarayanan et al. (2015)][])

### Quantitative Results

The following table outlines the testing results from SegNet.

| Metric      | Validation |
|:------------|:-----------|
| Accuracy    | 0.846443
| mean IoU    | 0.447931
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


<!--
## [Bayesian SegNet][Kendall et al. (2015)]

-   encoder transfer learning from VGG16 trained on ImageNet
-   optimized using SGD with 𝛃=0.9, initial α=0.001
    -   learning rate decay of 0.95 per epoch
-   trained for for max of 200 epochs with batch size 8
    -   training stopped early based on training loss with patience 10
-   best model in terms of training loss is kept as final model
-   median frequency balancing of class labels ([Eigen et al. (2014)][])
    -   weighted categorical cross-entropy loss function
-   local contrast normalization of inputs ([LeCun et al. (2009)][])
-   pooling indexes ([Badrinarayanan et al. (2015)][])
-   dropout rate of 50%
-   40 Monte Carlo samples to estimate mean class and variance

### Quantitative Results

The following table outlines the testing results from SegNet.

### Qualitative Results
-->

<!--
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

-   trained on 224 x 224 for for 50 epochs with batch size 4 (patience 100)
    -   optimized using RMSprop with learning rate 0.001, decay 0.995
-   trained on 352 (360) x 480 for for 50 epochs with batch size 1 (patience 50)
    -   optimized using RMSprop with learning rate 0.001, decay 0.995
-   median frequency balancing of class labels ([Eigen et al. (2014)][])
    -   weighted categorical cross-entropy loss function
-   local contrast normalization of inputs ([LeCun et al. (2009)][])
-   skip connections ([Jégou et al. (2016)][])

### Quantitative Results

The following table outlines the testing results from 103 Layers Tiramisu.

### Qualitative Results
-->

<!--
## [Bayesian Tiramisu][Kendall et al. (2017)]

### Quantitative Results

The following table outlines the testing results from SegNet.

### Qualitative Results
-->

<!-- References -->

[LeCun et al. (2009)]: http://yann.lecun.com/exdb/publis/pdf/jarrett-iccv-09.pdf
[Eigen et al. (2014)]: https://arxiv.org/abs/1411.4734
[Badrinarayanan et al. (2015)]: https://arxiv.org/pdf/1511.00561.pdf
[Kendall et al. (2015)]: https://arxiv.org/abs/1511.02680
[Jégou et al. (2016)]: https://arxiv.org/abs/1611.09326
[Kendall et al. (2017)]: http://papers.nips.cc/paper/7141-what-uncertainties-do-we-need-in-bayesian-deep-learning-for-computer-vision
