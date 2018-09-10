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
in the original paper. The first 32 table entries represent the final IoU 
for a given class label. The last two entries are general accuracy and mean 
IoU.

<table>
  <thead>
    <tr>
      <th></th>
      <th>train</th>
      <th>val</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Animal</th>
      <td>0.975410</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Archway</th>
      <td>0.889344</td>
      <td>0.942857</td>
    </tr>
    <tr>
      <th>Bicyclist</th>
      <td>0.113073</td>
      <td>0.185735</td>
    </tr>
    <tr>
      <th>Bridge</th>
      <td>1.000000</td>
      <td>0.957143</td>
    </tr>
    <tr>
      <th>Building</th>
      <td>0.617661</td>
      <td>0.313628</td>
    </tr>
    <tr>
      <th>Car</th>
      <td>0.256293</td>
      <td>0.152737</td>
    </tr>
    <tr>
      <th>CartLuggagePram</th>
      <td>0.864754</td>
      <td>0.857143</td>
    </tr>
    <tr>
      <th>Child</th>
      <td>0.784844</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Column_Pole</th>
      <td>0.065878</td>
      <td>0.052255</td>
    </tr>
    <tr>
      <th>Fence</th>
      <td>0.107964</td>
      <td>0.105737</td>
    </tr>
    <tr>
      <th>LaneMkgsDriv</th>
      <td>0.416986</td>
      <td>0.112596</td>
    </tr>
    <tr>
      <th>LaneMkgsNonDriv</th>
      <td>0.950820</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Misc_Text</th>
      <td>0.111666</td>
      <td>0.296739</td>
    </tr>
    <tr>
      <th>MotorcycleScooter</th>
      <td>0.975410</td>
      <td>0.971429</td>
    </tr>
    <tr>
      <th>OtherMoving</th>
      <td>0.061881</td>
      <td>0.431455</td>
    </tr>
    <tr>
      <th>ParkingBlock</th>
      <td>0.643516</td>
      <td>0.714286</td>
    </tr>
    <tr>
      <th>Pedestrian</th>
      <td>0.067082</td>
      <td>0.287478</td>
    </tr>
    <tr>
      <th>Road</th>
      <td>0.693613</td>
      <td>0.306640</td>
    </tr>
    <tr>
      <th>RoadShoulder</th>
      <td>0.840164</td>
      <td>0.957143</td>
    </tr>
    <tr>
      <th>SUVPickupTruck</th>
      <td>0.772541</td>
      <td>0.414286</td>
    </tr>
    <tr>
      <th>Sidewalk</th>
      <td>0.508683</td>
      <td>0.064265</td>
    </tr>
    <tr>
      <th>SignSymbol</th>
      <td>0.325357</td>
      <td>0.842857</td>
    </tr>
    <tr>
      <th>Sky</th>
      <td>0.716677</td>
      <td>0.410243</td>
    </tr>
    <tr>
      <th>TrafficCone</th>
      <td>0.987705</td>
      <td>0.957143</td>
    </tr>
    <tr>
      <th>TrafficLight</th>
      <td>0.162510</td>
      <td>0.588398</td>
    </tr>
    <tr>
      <th>Train</th>
      <td>0.975410</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Tree</th>
      <td>0.314626</td>
      <td>0.366039</td>
    </tr>
    <tr>
      <th>Truck_Bus</th>
      <td>0.448216</td>
      <td>0.742857</td>
    </tr>
    <tr>
      <th>Tunnel</th>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>VegetationMisc</th>
      <td>0.262337</td>
      <td>0.488066</td>
    </tr>
    <tr>
      <th>Void</th>
      <td>0.023624</td>
      <td>0.009583</td>
    </tr>
    <tr>
      <th>Wall</th>
      <td>0.079310</td>
      <td>0.074076</td>
    </tr>
    <tr>
      <th>mean_iou</th>
      <td>0.531667</td>
      <td>0.550088</td>
    </tr>
    <tr>
      <th>acc</th>
      <td>0.760879</td>
      <td>0.494078</td>
    </tr>
  </tbody>
</table>

### Training

![train](https://user-images.githubusercontent.com/2184469/45189870-a8690880-b200-11e8-9b34-ae98fccd0e34.png)

### Validation

![validation](https://user-images.githubusercontent.com/2184469/45189872-aacb6280-b200-11e8-9597-030f6bccdf79.png)
