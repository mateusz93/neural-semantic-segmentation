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

[Train-Tiramisu103-CamVid11.ipynb](Train-Tiramisu103-CamVid11.ipynb) generates
these results for the 11 class version of CamVid defined in the mapping file
[11_class.txt](11_class.txt).

<!-- TODO: images -->

#### Metrics

| Metric          | Validation |
|:----------------|:-----------|
| acc             | 0.669579
| mean_iou        | 0.321581
| iou_Bicyclist   | 0.647619
| iou_Building    | 0.407584
| iou_Car         | 0.333378
| iou_Column_Pole | 0.098502
| iou_Fence       | 0.036177
| iou_Pedestrian  | 0.170564
| iou_Road        | 0.657058
| iou_Sidewalk    | 0.064504
| iou_SignSymbol  | 0.119745
| iou_Sky         | 0.497471
| iou_Tree        | 0.504785

### 32 Class

[Train-Tiramisu103-CamVid32.ipynb](Train-Tiramisu103-CamVid32.ipynb) generates
these results for the full 32 class version of CamVid.

<!-- TODO: images -->

#### Metrics

| Metric                  | Validation |
|:------------------------|:-----------|
| acc                     | 0.574637
| mean_iou                | 0.532636
| iou_Animal              | 0.995238
| iou_Archway             | 0.914286
| iou_Bicyclist           | 0.476190
| iou_Bridge              | 0.957143
| iou_Building            | 0.325219
| iou_Car                 | 0.221473
| iou_CartLuggagePram     | 0.619048
| iou_Child               | 0.971429
| iou_Column_Pole         | 0.129600
| iou_Fence               | 0.252261
| iou_LaneMkgsDriv        | 0.251791
| iou_LaneMkgsNonDriv     | 1.000000
| iou_Misc_Text           | 0.057317
| iou_MotorcycleScooter   | 0.971429
| iou_OtherMoving         | 0.355676
| iou_ParkingBlock        | 0.529081
| iou_Pedestrian          | 0.178085
| iou_Road                | 0.553283
| iou_RoadShoulder        | 0.900000
| iou_SUVPickupTruck      | 0.069617
| iou_Sidewalk            | 0.082412
| iou_SignSymbol          | 0.476190
| iou_Sky                 | 0.459560
| iou_TrafficCone         | 0.919048
| iou_TrafficLight        | 0.457983
| iou_Train               | 1.000000
| iou_Tree                | 0.476796
| iou_Truck_Bus           | 0.733333
| iou_Tunnel              | 1.000000
| iou_VegetationMisc      | 0.469454
| iou_Void                | 0.057503
| iou_Wall                | 0.183900
