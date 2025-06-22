"""Test for the Category class with a simple MNIST classification example."""

import torch

from mlcat.morphisms import ProductObject
from mlcat.categories import Category, Path, Equation


def test_category(setup_mnist_classifier):
    """Test the Category class with a simple MNIST classification example."""
    dataloader, classification_space, image_space, classifier = setup_mnist_classifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_space = ProductObject(
        image_space,
        classification_space,
        dataloaders={"MNIST": dataloader},
    )

    image_label_tuple = tuple(next(iter(train_space.dataloaders["MNIST"])))
    assert isinstance(image_label_tuple, tuple)
    assert len(image_label_tuple) == 2
    assert classification_space.loss_functions is not None
    loss = classification_space.loss_functions["CE"](
        classifier(train_space.projection_a((image_label_tuple))),
        train_space.projection_b(image_label_tuple),
    )
    assert isinstance(loss.item(), float)

    category = Category(
        morphisms=[
            classifier,
            train_space.projection_a,
            train_space.projection_b,
        ],
        device=device,
    )
    assert isinstance(classifier.func, torch.nn.Module)
    category.add_equation(
        Equation(
            path_1=Path(
                [train_space.projection_a, classifier],
                torch.optim.Adam(classifier.func.parameters(), lr=0.002),
            ),
            path_2=Path([train_space.projection_b], None),
            dataloader_name="MNIST",
            loss_function_name="CE",
        )
    )
