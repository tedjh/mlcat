"""A module for loading datasets and generating random images."""

from typing import Iterable

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms  # type: ignore


class CustomDataset(datasets.VisionDataset):
    """A custom dataset that assigns a fixed label to all images."""

    def __init__(self, dataset: datasets.VisionDataset, target: int):
        super().__init__()
        self.target = target
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index][0], self.target


class RandomImageGenerator:
    """A generator that produces random images of a specified shape and device."""

    def __init__(
        self,
        shape: torch.Size,
        device: torch.device,
        label: int | None = None,
    ):
        self.shape = shape
        self.device = device
        self.label = label

    def __iter__(self):
        return self

    def __next__(self):
        """Generate a random image tensor of the specified shape and device."""
        if self.label is None:
            return torch.randn(self.shape, device=self.device)

        return (
            torch.randn(self.shape, device=self.device),
            torch.full((self.shape[0],), self.label, device=self.device),
        )


def get_dataloader(
    dataset_name: str, fixed_label: int | None = None
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    """Get a DataLoader for the specified dataset with optional fixed labels."""
    if dataset_name == "MNIST":
        dataset_class = datasets.MNIST
        output_dim = 10
    elif dataset_name == "CIFAR":
        dataset_class = datasets.CIFAR10
        output_dim = 10
    else:
        raise NotImplementedError("Unrecognised dataset name")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )

    dataset = dataset_class(
        root="../data",
        train=True,
        download=True,
        transform=lambda x: transform(x).squeeze().flatten(),
        target_transform=(
            lambda x: torch.nn.functional.one_hot(  # pylint: disable=not-callable
                torch.Tensor([x]).long(), output_dim
            )
            .float()
            .squeeze()
        ),
    )
    if fixed_label is not None:
        dataset = CustomDataset(dataset, fixed_label)

    return DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
    )
