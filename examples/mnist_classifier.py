"""Example of a simple MNIST classifier using the mlcat library."""

import torch

from mlcat.objects import Object
from mlcat.morphisms import TorchMorphism, ProductObject
from mlcat.categories import Category, Path, Equation
from mlcat.data import get_dataloader


def main(dataset_name: str = "MNIST"):
    """
    Main function to set up and train a simple MNIST classifier using the mlcat library.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classification_space = Object(
        torch.Size([10]), {"CE": torch.nn.functional.cross_entropy}
    )
    image_space = Object(torch.Size([28 * 28]), None)

    train_space = ProductObject(
        image_space,
        classification_space,
        dataloaders={dataset_name: get_dataloader(dataset_name)},
    )

    classifier = TorchMorphism(
        image_space,
        classification_space,
        torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(28 * 28, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10),
        ),
    )
    assert isinstance(classifier.func, torch.nn.Module)
    category = Category(
        morphisms=[classifier, train_space.projection_a, train_space.projection_b],
        device=device,
        equations=[
            Equation(
                path_1=Path(
                    [train_space.projection_a, classifier],
                    torch.optim.Adam(classifier.func.parameters(), lr=0.001),
                ),
                path_2=Path([train_space.projection_b], None),
                dataloader_name="MNIST",
                loss_function_name="CE",
            )
        ],
    )
    category.train(10)


if __name__ == "__main__":
    main()
