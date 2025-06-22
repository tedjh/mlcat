"""Test cases for morphisms in the MLCat library."""

import torch

from mlcat.morphisms import TorchMorphism, ProductObject


def test_morphism_projection(setup_mnist_classifier):
    """Test the projection morphisms in a ProductObject."""
    dataloader, classification_space, image_space, _ = setup_mnist_classifier

    train_space = ProductObject(
        image_space,
        classification_space,
        dataloaders={"main": dataloader},
    )

    assert train_space.projection_a.dom == train_space
    assert train_space.projection_a.codom == image_space
    assert train_space.projection_b.dom == train_space
    assert train_space.projection_b.codom == classification_space

    test_tuple = next(iter(dataloader))

    torch.testing.assert_close(train_space.projection_a(test_tuple), test_tuple[0])
    torch.testing.assert_close(train_space.projection_b(test_tuple), test_tuple[1])


def test_morphism_torch(setup_mnist_classifier):
    """Test the TorchMorphism with a simple MNIST classification example."""
    dataloader, classification_space, _, classifier = setup_mnist_classifier
    test_tensor, test_label = next(iter(dataloader))
    assert classification_space.check_is_in(test_label)
    assert classification_space.check_is_in(classifier(test_tensor))


def test_morphism_calls(setup_mnist_classifier):
    """Test the TorchMorphism call and equality."""
    dataloader, classification_space, image_space, classifier = setup_mnist_classifier

    train_space = ProductObject(
        image_space,
        classification_space,
        dataloaders={"main": dataloader},
    )

    image_label_tuple = tuple(next(iter(train_space.dataloaders["main"])))
    assert classification_space.loss_functions is not None
    assert isinstance(image_label_tuple, tuple)
    assert len(image_label_tuple) == 2
    loss = classification_space.loss_functions["CE"](
        classifier(train_space.projection_a(image_label_tuple)),
        train_space.projection_b(image_label_tuple),
    )
    assert loss.item()
    # pylint: disable=duplicate-code
    classifier_2 = TorchMorphism(
        image_space,
        classification_space,
        torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(28 * 28, 15),
            torch.nn.ReLU(),
            torch.nn.Linear(15, 10),
        ),
    )

    assert classifier != classifier_2
