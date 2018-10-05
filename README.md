# Neural Semantic Segmentation

Implementations of neural network papers for semantic segmentation using Keras
and TensorFlow.

<p align="center">
  Predictions from Tiramisu on CamVid video stream.
  <img alt="Segmentation Demonstration" src="img/tiramisu/0005VD.gif" width="100%" />
</p>

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
    [11_class.txt](src/camvid/11_class.txt)
    -   use 12 labels and ignore the Void class (i.e., 11 labels)
-   960 x 720 scaled down by factor of 2 to 480 x 360

[CamVid]: http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/
[CamVid-classes]: http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/#ClassLabels

# Models

<details>
<summary>SegNet</summary>

## [SegNet][Badrinarayanan et al. 2015]

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
-   median frequency balancing of class labels ([Eigen et al. 2014][])
    -   weighted categorical cross-entropy loss function
-   local contrast normalization of inputs ([Jarrett et al. 2009][])
-   pooling indexes ([Badrinarayanan et al. 2015][])

### Quantitative Results

The following table outlines the testing results from SegNet.

| Metric                  |    Value |
|:------------------------|---------:|
| Accuracy                | 0.885892 |
| Mean Per Class Accuracy | 0.725983 |
| Mean I/U                | 0.565462 |
| Bicyclist               | 0.424479 |
| Building                | 0.735429 |
| Car                     | 0.685271 |
| Column_Pole             | 0.253229 |
| Fence                   | 0.317258 |
| Pedestrian              | 0.372688 |
| Road                    | 0.905312 |
| Sidewalk                | 0.772007 |
| SignSymbol              | 0.180342 |
| Sky                     | 0.889403 |
| VegetationMisc          | 0.684661 |

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

</details>



<details>
<summary>Bayesian SegNet</summary>

## [Bayesian SegNet][Kendall et al. 2015]

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
-   median frequency balancing of class labels ([Eigen et al. 2014][])
    -   weighted categorical cross-entropy loss function
-   local contrast normalization of inputs ([Jarrett et al. 2009][])
-   pooling indexes ([Badrinarayanan et al. 2015][])

### Quantitative Results

The following table outlines the testing results from Bayesian SegNet.

| Metric                  |    Value |
|:------------------------|---------:|
| Accuracy                | 0.868758 |
| Mean Per Class Accuracy | 0.767013 |
| Mean I/U                | 0.550385 |
| Bicyclist               | 0.369756 |
| Building                | 0.709912 |
| Car                     | 0.675461 |
| Column_Pole             | 0.20731  |
| Fence                   | 0.394083 |
| Pedestrian              | 0.326561 |
| Road                    | 0.885637 |
| Sidewalk                | 0.72415  |
| SignSymbol              | 0.174897 |
| Sky                     | 0.886314 |
| VegetationMisc          | 0.700158 |

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

</details>



<details>
<summary>The One Hundred Layers Tiramisu</summary>

## [The One Hundred Layers Tiramisu][Jégou et al. 2016]

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

-   random _horizontal_ flips of images during training
    -   the paper says vertical, but their implementation clearly shows
        horizontal flips (likely a typo). Horizontal make more sense than
        vertical anyway and produces empirically better test results
-   batch normalization statistics computed _per batch_ during training,
    validation, and testing
-   skip connections between encoder and decoder ([Jégou et al. 2016][])

### Quantitative Results

The following table outlines the testing results from 103 Layers Tiramisu.

| Metric                  |    Value |
|:------------------------|---------:|
| Accuracy                | 0.901727 |
| Mean Per Class Accuracy | 0.685065 |
| Mean I/U                | 0.567453 |
| Bicyclist               | 0.332004 |
| Building                | 0.770349 |
| Car                     | 0.668687 |
| Column_Pole             | 0.257645 |
| Fence                   | 0.247566 |
| Pedestrian              | 0.429406 |
| Road                    | 0.901876 |
| Sidewalk                | 0.769319 |
| SignSymbol              | 0.22198  |
| Sky                     | 0.921203 |
| VegetationMisc          | 0.721953 |

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

</details>



<details>
<summary>Bayesian The One Hundred Layers Tiramisu</summary>

## [Bayesian Tiramisu][Kendall et al. 2017]

### Aleatoric Uncertainty

![Aleatoric Tiramisu](img/bayesian-tiramisu/aleatoric/arch.png)

The following table describes training hyperparameters.

| Crop Size | Epochs | Batch Size | Patience | Optimizer | α    | α Decay | Dropout |
|:----------|:-------|:-----------|:---------|:----------|:-----|:--------|:--------|
| 352 x 480 | 100    | 1          | 10       | RMSprop   | 1e-4 | 1.000   | 20%     |

-   network split to predict targets and loss attenuation
    -   custom loss function to train the second head of the network
        ([Kendall et al. 2017][])
    -   our loss function samples _through the Softmax function_ like their
        paper says (but contrary to the mathematics they present?). without
        applying the Softmax function, the loss is unstable and goes negative
-   pre-trained with fine weights from original Tiramisu
-   pre-trained network frozen while head to predict sigma is trained

#### Quantitative Results

The quantitative results are the same as the standard Tiramisu model.

#### Qualitative Results

<table>
  <tr>
    <td>
      <img src="img/bayesian-tiramisu/aleatoric/0.png" />
    </td>
    <td>
      <img src="img/bayesian-tiramisu/aleatoric/1.png" />
    </td>
    <td>
      <img src="img/bayesian-tiramisu/aleatoric/2.png" />
    </td>
    <td>
      <img src="img/bayesian-tiramisu/aleatoric/3.png" />
    </td>
  </tr>
</table>

### Epistemic Uncertainty

-   pre-trained with fine weights from original Tiramisu
-   50 samples for Monte Carlo Dropout sampling at test time

#### Quantitative Results

The following table outlines the testing results from Epistemic Tiramisu.

<!-- TODO: update with new values -->

| Metric                  |    Value |
|:------------------------|---------:|
| Accuracy                | 0.906531 |
| Mean Per Class Accuracy | 0.696133 |
| Mean I/U                | 0.573954 |
| Bicyclist               | 0.361406 |
| Building                | 0.773772 |
| Car                     | 0.660941 |
| Column/Pole             | 0.288474 |
| Fence                   | 0.188135 |
| Pedestrian              | 0.444522 |
| Road                    | 0.914459 |
| Sidewalk                | 0.804674 |
| Sign                    | 0.226198 |
| Sky                     | 0.923827 |
| Vegetation              | 0.727091 |

#### Qualitative Results

<table>
  <tr>
    <td>
      <img src="img/bayesian-tiramisu/epistemic/0.png" />
    </td>
    <td>
      <img src="img/bayesian-tiramisu/epistemic/1.png" />
    </td>
    <td>
      <img src="img/bayesian-tiramisu/epistemic/2.png" />
    </td>
    <td>
      <img src="img/bayesian-tiramisu/epistemic/3.png" />
    </td>
  </tr>
</table>

</details>



<details>
<summary>Wall Clock Inference Time Metrics</summary>

## Wall Clock Inference Time Metrics

The following box plot describes the mean and standard deviation in wall clock
time execution of different segmentation models performing inference on images
of size 352 x 480 pixels.

![Wall Clock Inference Times (Deterministic Inference)](img/inference-time.png)

The following box plot describes the mean and standard deviation in wall clock
time execution of different Bayesian segmentation models performing inference
on images of size 352 x 480 pixels. Note that in this case, inference is
probabilistic due to the test time dropout and Monte Carlo simulation over 50
network samples.

![Wall Clock Inference Times (Monte Carlo Dropout)](img/inference-time-mc-dropout.png)

</details>



# References

[Badrinarayanan V, Kendall A, Cipolla R (2015) SegNet: A Deep Convolutional Encoder-Decoder Architec- ture for Image Segmentation. ArXiv e-prints.][Badrinarayanan et al. 2015]

[Eigen D, Fergus R (2014) Predicting Depth, Surface Normals and Semantic Labels with a Common Multi- Scale Convolutional Architecture. ArXiv e-prints.][Eigen et al. 2014]

[Jarrett K, Kavukcuoglu K, Ranzato M, LeCun Y (2009) What is the best multi-stage architecture for object recognition? 2009 IEEE 12th International Conference on Computer Vision, 2146–2153.][Jarrett et al. 2009]

[Jégou S, Drozdzal M, Vazquez D, Romero A, Bengio Y (2016) The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation. ArXiv e-prints.][Jégou et al. 2016]

[Kendall A, Badrinarayanan V, Cipolla R (2015) Bayesian SegNet: Model Uncertainty in Deep Convolutional Encoder-Decoder Architectures for Scene Understanding. ArXiv e-prints.][Kendall et al. 2015]

[Kendall A, Gal Y (2017) What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? ArXiv e-prints.][Kendall et al. 2017]

[Badrinarayanan et al. 2015]: https://arxiv.org/abs/1511.00561
[Eigen et al. 2014]: https://arxiv.org/abs/1411.4734
[Jarrett et al. 2009]: https://ieeexplore.ieee.org/document/5459469
[Jégou et al. 2016]: https://arxiv.org/abs/1611.09326
[Kendall et al. 2015]: https://arxiv.org/abs/1511.02680
[Kendall et al. 2017]: https://arxiv.org/abs/1703.04977
