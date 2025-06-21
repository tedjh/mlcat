import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pytest

from mlcat.objects import Object
from mlcat.morphisms import TorchMorphism, ProductObject


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
            download=True,
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


def test_morphism_projection(dataloader):
    classification_space = Object(
        torch.Size([10]), {"CE": torch.nn.functional.cross_entropy}
    )
    image_space = Object(torch.Size([28, 28]), None)

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


def test_morphism_torch(dataloader):
    classification_space = Object(
        torch.Size([10]), {"CE": torch.nn.functional.cross_entropy}
    )
    image_space = Object(torch.Size([28, 28]), None)

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
    test_tensor, test_label = next(iter(dataloader))
    assert classification_space.check_is_in(test_label)
    assert classification_space.check_is_in(classifier(test_tensor))


def test_morphism_calls(dataloader):
    classification_space = Object(
        torch.Size([10]), {"CE": torch.nn.functional.cross_entropy}
    )
    image_space = Object(torch.Size([28, 28]), None)

    train_space = ProductObject(
        image_space,
        classification_space,
        dataloaders={"main": dataloader},
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

    image_label_tuple = next(iter(train_space.dataloaders["main"]))
    loss = classification_space.loss_functions["CE"](
        classifier(train_space.projection_a(image_label_tuple)),
        train_space.projection_b(image_label_tuple),
    )
    assert loss.item()

    classifier_2 = TorchMorphism(
        image_space,
        classification_space,
        torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(28 * 28, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10),
        ),
    )

    assert classifier != classifier_2
