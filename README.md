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

The following table outlines the testing results from SegNet on CamVid 11.

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

<!-- ### 11 Class

[Train-Tiramisu103-CamVid11.ipynb](Train-Tiramisu103-CamVid11.ipynb) generates
these results for the 11 class version of CamVid defined in the mapping file
[11_class.txt](11_class.txt).

#### Metrics

| Metric          | Validation |
|:----------------|:-----------|
| acc             | 0.669579
| mean_iou        | 0.321581
| Bicyclist   | 0.647619
| Building    | 0.407584
| Car         | 0.333378
| Column_Pole | 0.098502
| Fence       | 0.036177
| Pedestrian  | 0.170564
| Road        | 0.657058
| Sidewalk    | 0.064504
| SignSymbol  | 0.119745
| Sky         | 0.497471
| Tree        | 0.504785

### 32 Class

[Train-Tiramisu103-CamVid32.ipynb](Train-Tiramisu103-CamVid32.ipynb) generates
these results for the full 32 class version of CamVid.

#### Metrics

| Metric                  | Validation |
|:------------------------|:-----------|
| acc                     | 0.574637
| mean_iou                | 0.532636
| Animal              | 0.995238
| Archway             | 0.914286
| Bicyclist           | 0.476190
| Bridge              | 0.957143
| Building            | 0.325219
| Car                 | 0.221473
| CartLuggagePram     | 0.619048
| Child               | 0.971429
| Column_Pole         | 0.129600
| Fence               | 0.252261
| LaneMkgsDriv        | 0.251791
| LaneMkgsNonDriv     | 1.000000
| Misc_Text           | 0.057317
| MotorcycleScooter   | 0.971429
| OtherMoving         | 0.355676
| ParkingBlock        | 0.529081
| Pedestrian          | 0.178085
| Road                | 0.553283
| RoadShoulder        | 0.900000
| SUVPickupTruck      | 0.069617
| Sidewalk            | 0.082412
| SignSymbol          | 0.476190
| Sky                 | 0.459560
| TrafficCone         | 0.919048
| TrafficLight        | 0.457983
| Train               | 1.000000
| Tree                | 0.476796
| Truck_Bus           | 0.733333
| Tunnel              | 1.000000
| VegetationMisc      | 0.469454
| Void                | 0.057503
| Wall                | 0.183900

-->
