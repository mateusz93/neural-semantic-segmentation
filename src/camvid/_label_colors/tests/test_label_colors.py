"""Test cases for the label colors module."""
from unittest import TestCase
import os
import glob
import numpy as np
from matplotlib import pyplot as plt


# the path to this directory
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# the directory containing the CIFAR-10 batches
IMGS = sorted(glob.glob(os.path.join(THIS_DIR, 'imgs/*.png')))


class ShouldMapDiscreteNonVectorized(TestCase):
    def test(self):
        from ..segmentation_to_discrete import SegmentationToDiscreteTransformer
        transformer = SegmentationToDiscreteTransformer()
        img = (255 * plt.imread(IMGS[0])).astype('uint8')
        test = transformer.unmap(transformer.map(img))
        self.assertTrue(np.array_equal(img, test))


class ShouldMapDiscreteVectorized(TestCase):
    def test(self):
        from ..segmentation_to_discrete import SegmentationToDiscreteTransformer
        transformer = SegmentationToDiscreteTransformer()
        img = np.array([
            (255 * plt.imread(IMGS[0])).astype('uint8'),
            (255 * plt.imread(IMGS[1])).astype('uint8'),
        ])
        test = transformer.unmap(transformer.map(img))
        self.assertTrue(np.array_equal(img, test))


class ShouldMapOneHotNonVectorized(TestCase):
    def test(self):
        from ..segmentation_to_onehot import SegmentationToOnehotTransformer
        transformer = SegmentationToOnehotTransformer()
        img = (255 * plt.imread(IMGS[0])).astype('uint8')
        test = transformer.unmap(transformer.map(img))
        self.assertTrue(np.array_equal(img, test))


class ShouldMapOneHotVectorized(TestCase):
    def test(self):
        from ..segmentation_to_onehot import SegmentationToOnehotTransformer
        transformer = SegmentationToOnehotTransformer()
        img = np.array([
            (255 * plt.imread(IMGS[0])).astype('uint8'),
            (255 * plt.imread(IMGS[1])).astype('uint8'),
        ])
        test = transformer.unmap(transformer.map(img))
        self.assertTrue(np.array_equal(img, test))
