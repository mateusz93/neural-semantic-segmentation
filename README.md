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
in the original paper. Results were generated using a machine equipped with 
128GB RAM, nVidia P100 GPU, and Intel Xeon CPU @ 2.10GHz. 

### Coarse Crops

results from training on random crops and flips of size 224x224.

#### Training Sample

![coarse-train](https://user-images.githubusercontent.com/2184469/45374681-6f41e700-b5b8-11e8-94d5-5fbcb534072e.png)

#### Validation Sample

![coarse-validation](https://user-images.githubusercontent.com/2184469/45374682-6f41e700-b5b8-11e8-9e20-9f1565e67029.png)

#### Metrics

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
      <th>acc</th>
      <td>0.728145</td>
      <td>0.476104</td>
    </tr>
    <tr>
      <th>Animal</th>
      <td>0.981557</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Archway</th>
      <td>0.907787</td>
      <td>0.914286</td>
    </tr>
    <tr>
      <th>Bicyclist</th>
      <td>0.107173</td>
      <td>0.185965</td>
    </tr>
    <tr>
      <th>Bridge</th>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Building</th>
      <td>0.570667</td>
      <td>0.297297</td>
    </tr>
    <tr>
      <th>Car</th>
      <td>0.196439</td>
      <td>0.137479</td>
    </tr>
    <tr>
      <th>CartLuggagePram</th>
      <td>0.846311</td>
      <td>0.828571</td>
    </tr>
    <tr>
      <th>Child</th>
      <td>0.747984</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Column_Pole</th>
      <td>0.036941</td>
      <td>0.036962</td>
    </tr>
    <tr>
      <th>Fence</th>
      <td>0.087507</td>
      <td>0.828571</td>
    </tr>
    <tr>
      <th>LaneMkgsDriv</th>
      <td>0.358721</td>
      <td>0.084688</td>
    </tr>
    <tr>
      <th>LaneMkgsNonDriv</th>
      <td>0.981557</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Misc_Text</th>
      <td>0.223213</td>
      <td>0.342857</td>
    </tr>
    <tr>
      <th>MotorcycleScooter</th>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>OtherMoving</th>
      <td>0.283316</td>
      <td>0.742857</td>
    </tr>
    <tr>
      <th>ParkingBlock</th>
      <td>0.639471</td>
      <td>0.742857</td>
    </tr>
    <tr>
      <th>Pedestrian</th>
      <td>0.066807</td>
      <td>0.244111</td>
    </tr>
    <tr>
      <th>Road</th>
      <td>0.637366</td>
      <td>0.377883</td>
    </tr>
    <tr>
      <th>RoadShoulder</th>
      <td>0.803310</td>
      <td>0.957143</td>
    </tr>
    <tr>
      <th>SUVPickupTruck</th>
      <td>0.645535</td>
      <td>0.428571</td>
    </tr>
    <tr>
      <th>Sidewalk</th>
      <td>0.364819</td>
      <td>0.067987</td>
    </tr>
    <tr>
      <th>SignSymbol</th>
      <td>0.508197</td>
      <td>0.842857</td>
    </tr>
    <tr>
      <th>Sky</th>
      <td>0.649620</td>
      <td>0.342699</td>
    </tr>
    <tr>
      <th>TrafficCone</th>
      <td>0.993852</td>
      <td>0.957143</td>
    </tr>
    <tr>
      <th>TrafficLight</th>
      <td>0.102376</td>
      <td>0.403704</td>
    </tr>
    <tr>
      <th>Train</th>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Tree</th>
      <td>0.319282</td>
      <td>0.340358</td>
    </tr>
    <tr>
      <th>Truck_Bus</th>
      <td>0.284645</td>
      <td>0.685714</td>
    </tr>
    <tr>
      <th>Tunnel</th>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>VegetationMisc</th>
      <td>0.244620</td>
      <td>0.494287</td>
    </tr>
    <tr>
      <th>Void</th>
      <td>0.007622</td>
      <td>0.004440</td>
    </tr>
    <tr>
      <th>Wall</th>
      <td>0.105014</td>
      <td>0.115342</td>
    </tr>
    <tr>
      <th>loss</th>
      <td>1.121514</td>
      <td>2.072659</td>
    </tr>
    <tr>
      <th>mean_iou</th>
      <td>0.521928</td>
      <td>0.575145</td>
    </tr>
  </tbody>
</table>

### Fine Tuning

Fine tuning on larger images size 512x640.

#### Training Sample

![fine-training](https://user-images.githubusercontent.com/2184469/45374683-6f41e700-b5b8-11e8-840c-a2736f2d5b58.png)

#### Fine Validation Sample

![fine-validation](https://user-images.githubusercontent.com/2184469/45374685-6f41e700-b5b8-11e8-84f3-422747af89a3.png)

#### Metrics

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
      <th>acc</th>
      <td>0.896648</td>
      <td>0.546369</td>
    </tr>
    <tr>
      <th>Animal</th>
      <td>0.963340</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Archway</th>
      <td>0.798506</td>
      <td>0.942857</td>
    </tr>
    <tr>
      <th>Bicyclist</th>
      <td>0.306020</td>
      <td>0.220589</td>
    </tr>
    <tr>
      <th>Bridge</th>
      <td>0.995927</td>
      <td>0.961905</td>
    </tr>
    <tr>
      <th>Building</th>
      <td>0.826583</td>
      <td>0.310521</td>
    </tr>
    <tr>
      <th>Car</th>
      <td>0.625598</td>
      <td>0.203755</td>
    </tr>
    <tr>
      <th>CartLuggagePram</th>
      <td>0.718941</td>
      <td>0.676190</td>
    </tr>
    <tr>
      <th>Child</th>
      <td>0.595707</td>
      <td>0.985714</td>
    </tr>
    <tr>
      <th>Column_Pole</th>
      <td>0.307140</td>
      <td>0.125378</td>
    </tr>
    <tr>
      <th>Fence</th>
      <td>0.389649</td>
      <td>0.424261</td>
    </tr>
    <tr>
      <th>LaneMkgsDriv</th>
      <td>0.602786</td>
      <td>0.217223</td>
    </tr>
    <tr>
      <th>LaneMkgsNonDriv</th>
      <td>0.957230</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Misc_Text</th>
      <td>0.224218</td>
      <td>0.198917</td>
    </tr>
    <tr>
      <th>MotorcycleScooter</th>
      <td>0.991853</td>
      <td>0.971429</td>
    </tr>
    <tr>
      <th>OtherMoving</th>
      <td>0.230130</td>
      <td>0.361124</td>
    </tr>
    <tr>
      <th>ParkingBlock</th>
      <td>0.324170</td>
      <td>0.676339</td>
    </tr>
    <tr>
      <th>Pedestrian</th>
      <td>0.319593</td>
      <td>0.144022</td>
    </tr>
    <tr>
      <th>Road</th>
      <td>0.931570</td>
      <td>0.374222</td>
    </tr>
    <tr>
      <th>RoadShoulder</th>
      <td>0.663167</td>
      <td>0.942857</td>
    </tr>
    <tr>
      <th>SUVPickupTruck</th>
      <td>0.200876</td>
      <td>0.084495</td>
    </tr>
    <tr>
      <th>Sidewalk</th>
      <td>0.769941</td>
      <td>0.067508</td>
    </tr>
    <tr>
      <th>SignSymbol</th>
      <td>0.215201</td>
      <td>0.353555</td>
    </tr>
    <tr>
      <th>Sky</th>
      <td>0.918880</td>
      <td>0.523964</td>
    </tr>
    <tr>
      <th>TrafficCone</th>
      <td>0.991853</td>
      <td>0.947619</td>
    </tr>
    <tr>
      <th>TrafficLight</th>
      <td>0.387686</td>
      <td>0.379938</td>
    </tr>
    <tr>
      <th>Train</th>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Tree</th>
      <td>0.627418</td>
      <td>0.507400</td>
    </tr>
    <tr>
      <th>Truck_Bus</th>
      <td>0.361422</td>
      <td>0.528571</td>
    </tr>
    <tr>
      <th>Tunnel</th>
      <td>0.997963</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>VegetationMisc</th>
      <td>0.257634</td>
      <td>0.305221</td>
    </tr>
    <tr>
      <th>Void</th>
      <td>0.274630</td>
      <td>0.036130</td>
    </tr>
    <tr>
      <th>Wall</th>
      <td>0.376459</td>
      <td>0.211167</td>
    </tr>
    <tr>
      <th>loss</th>
      <td>0.441265</td>
      <td>2.002783</td>
    </tr>
    <tr>
      <th>mean_iou</th>
      <td>0.598503</td>
      <td>0.521340</td>
    </tr>
  </tbody>
</table>
