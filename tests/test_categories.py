import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from mlcat.objects import Object
from mlcat.morphisms import TorchMorphism, ProductObject
from mlcat.categories import Category, TrainPath, Equation
import pytest


@pytest.fixture
def dataloader():
    batch_size = 64
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]  # scale to [-1, 1]
    )
    return DataLoader(
        datasets.MNIST(
            root="./data",
            train=True,
            download=False,
            transform=lambda x: transform(x).squeeze(),
            target_transform=lambda x: torch.nn.functional.one_hot(
                torch.Tensor([x]).long(), 10
            )
            .float()
            .squeeze(),
        ),
        batch_size=batch_size,
        shuffle=True,
    )


def test_category(dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classification_space = Object(
        torch.Size([10]), {"CE": torch.nn.functional.cross_entropy}
    )
    image_space = Object(torch.Size([28, 28]), None)

    train_space = ProductObject(
        image_space,
        classification_space,
        dataloaders={"MNIST": dataloader},
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

    image_label_tuple = next(iter(train_space.dataloaders["MNIST"]))
    loss = classification_space.loss_functions["CE"](
        classifier(train_space.projection_a(image_label_tuple)),
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
            path_1=TrainPath(
                [train_space.projection_a, classifier],
                torch.optim.Adam(classifier.func.parameters(), lr=0.001),
            ),
            path_2=TrainPath([train_space.projection_b], None),
            dataloader_name="MNIST",
            loss_function_name="CE",
        )
    )
