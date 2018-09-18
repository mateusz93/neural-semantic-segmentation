# CamVid

## `CamVid` class

`CamVid` works like a Keras DataGenerator to load the dataset dynamically at
runtime training of the model.

```python
import camvid
camvid32 = camvid.CamVid()
generators = camvid32.generators()
```

generators is a dictionary with values for 'training' and 'validation' sets
respectively.

### <32 Class

To generate the new label data using an alternate mapping, supply a dictionary
of replacements to the mapping keyword:

```python
import camvid
camvid.CamVid(mapping={'Animal': 'Void'})
```

## `plot` method

`plot` take any number of keyword arguments where the key indicates the name
of the image and the value is a NumPy tensor of pixel data.

```python
import camvid
# load the 32 label dataset
camvid32 = camvid.CamVid()
# create a generator for the data
generators = camvid32.generators()
# generate a batch of 3 images from the training set
X, y = next(generators['training'])
# unmap the discrete onehot tensor into an RGB image
y = camvid32.unmap(y)
# plot the data (first image)
camvid.plot(
    X=X[0],
    y=y[0],
)
```
