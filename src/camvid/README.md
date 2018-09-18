# CamVid

## `create_segmented_y` method

Before initializing an instance of `CamVid`, a preprocessed version of the
target images much be generated. Depending on the number of labels in the
output, the method will generate a directory `y_<number of labels>` with the
preprocessed data.

### 32 Class

To generate the standard 32 class dataset labels:

```python
import camvid
camvid.create_segmented_y()
```

The labels follow the distribution:

![DataPercents](DataPercents.jpg)

### <32 Class

To generate the new label data using an alternate mapping, supply a dictionary
of replacements to the mapping keyword:

```python
import camvid
camvid.create_segmented_y(mapping={'Animal': 'Void'})
```

## `CamVid` class

`CamVid` works like a Keras DataGenerator to load the dataset dynamically at
runtime training of the model.

```python
import camvid
camvid32 = camvid.CamVid('y_32')
generators = camvid32.generators()
```

generators is a dictionary with values for 'training' and 'validation' sets
respectively.

## `plot` method

`plot` take any number of keyword arguments where the key indicates the name
of the image and the value is a NumPy tensor of pixel data.

```python
import camvid
camvid32 = camvid.CamVid('y_32')
generators = camvid32.generators()
X, y = next(generators['training'])
y = camvid32.unmap(y)
camvid.plot(
    X=...,
    y=...,
)
```
