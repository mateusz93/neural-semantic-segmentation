# The 100 Layers Tiramisu (Implementation)

An Implementation of
_[The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation][100-layer-tiramisu]_.

[100-layer-tiramisu]: papers/the-100-layers-tiramisu.pdf

## Installation

To install requirements for the project:

```shell
python -m pip install -r requirements.txt
```

## Usage

[Train-Tiramisu103-CamVid.ipynb](Train-Tiramisu103-CamVid.ipynb) contains
logic to train and validate the 103 layers Tiramisu on both crops and full
size images.

## Results

Note that results are for the 32 class CamVid dataset, not the 11 class used
in the original paper.

### Training

![train](https://user-images.githubusercontent.com/2184469/45189870-a8690880-b200-11e8-9b34-ae98fccd0e34.png)

### Validation

![validation](https://user-images.githubusercontent.com/2184469/45189872-aacb6280-b200-11e8-9597-030f6bccdf79.png)
