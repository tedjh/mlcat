import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pytest

from mlcat.objects import Object, ObjectError
from mlcat.morphisms import ProductObject


@pytest.fixture
def dataloader():
    batch_size = 64
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )
    return DataLoader(
        datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=lambda x: transform(x).squeeze(),
            target_transform=lambda x: torch.nn.functional.one_hot(
                torch.Tensor([x]).long(), 10
            ).squeeze(),
        ),
        batch_size=batch_size,
        shuffle=True,
    )


def test_object(dataloader):
    out_space = Object(torch.Size([10]), {"CE": torch.nn.functional.cross_entropy})
    assert out_space.shape == torch.Size([10])
    assert out_space.loss_functions["CE"] == torch.nn.functional.cross_entropy

    image_space = Object(torch.Size([28, 28]), None)
    assert image_space.loss_functions is None

    assert out_space != image_space
    assert out_space == Object(torch.Size([10]), None)

    assert out_space.check_is_in(torch.Tensor(range(10)))
    assert not out_space.check_is_in(torch.randn(torch.Size([28, 28])))


def test_product_object(dataloader):
    out_space = Object(torch.Size([10]), {"CE": torch.nn.functional.cross_entropy})
    image_space = Object(torch.Size([28, 28]), None)
    train_space = ProductObject(
        image_space,
        out_space,
        dataloaders={"main": dataloader},
    )
    assert train_space.a == image_space
    assert train_space.b == out_space
    assert train_space.dataloaders["main"] == dataloader

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),  # std
        ]
    )
    cifar_loader = DataLoader(
        datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transform,
            target_transform=lambda x: torch.nn.functional.one_hot(
                torch.Tensor([x]).long(), 10
            ).squeeze(),
        ),
        batch_size=64,
        shuffle=True,
    )
    with pytest.raises(ObjectError):
        ProductObject(
            image_space,
            out_space,
            dataloaders={"main": cifar_loader},
        )
