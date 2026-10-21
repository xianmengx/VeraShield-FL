"""Dataset/model contract tests; real-data convergence is a separate smoke run."""

import sys as _release_sys
from pathlib import Path as _ReleasePath
_RELEASE_ROOT = _ReleasePath(__file__).resolve().parents[1]
import numpy as np
import pytest
import torch
from PIL import Image

from pytorchexample.task import Net, apply_mnist_transforms, apply_transforms, add_backdoor_trigger


@pytest.mark.parametrize("dataset,channels,size,features", [("cifar10", 3, 32, 400), ("mnist", 1, 28, 256)])
def test_model_shapes_and_serialization(dataset, channels, size, features):
    net = Net(dataset)
    other = Net(dataset)
    other.load_state_dict(net.state_dict())
    assert net.fc1.in_features == features
    images = torch.randn(2, channels, size, size)
    assert net(images).shape == (2, 10)
    assert torch.equal(net(images), other(images))
    assert add_backdoor_trigger(images).shape == images.shape


def test_dataset_transforms_use_common_batch_contract():
    mnist = apply_mnist_transforms({"image": [Image.fromarray(np.zeros((28, 28), dtype=np.uint8))], "label": [0]})
    cifar = apply_transforms({"img": [Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))], "label": [0]})
    assert mnist["img"][0].shape == (1, 28, 28)
    assert cifar["img"][0].shape == (3, 32, 32)
    assert mnist["img"][0].min() == -1
    assert mnist["label"] == [0]


def test_unknown_dataset_is_rejected():
    with pytest.raises(ValueError, match="Unsupported dataset"):
        Net("unknown")
