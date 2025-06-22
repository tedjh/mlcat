"""Configuration for pytest fixtures and hooks."""

import pytest
import torch

from mlcat.objects import Object
from mlcat.morphisms import TorchMorphism
from mlcat.data import get_dataloader


@pytest.fixture
def setup_mnist_classifier():
    """Fixture to set up a simple MNIST classifier."""
    dataloader = get_dataloader("MNIST")
    classification_space = Object(
        torch.Size([10]), {"CE": torch.nn.functional.cross_entropy}
    )
    image_space = Object(torch.Size([28 * 28]), None)
    # pylint: disable=duplicate-code
    classifier = TorchMorphism(
        image_space,
        classification_space,
        torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(28 * 28, 15),
            torch.nn.ReLU(),
            torch.nn.Linear(15, 10),
        ),
    )

    return dataloader, classification_space, image_space, classifier
