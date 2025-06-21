import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class CustomDataset:
    def __init__(self, dataset, target: int, *args, **kwargs):
        self.target = target
        self.dataset = dataset
        super().__init__(*args, **kwargs)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[0], self.target


class RandomGenerator:
    def __init__(
        self,
        shape: torch.Size,
        device: torch.device,
        label: int | None = None,
    ):
        self.shape = shape
        self.device = device
        self.label = label

    def __iter__(self):  # reset iteration
        return self

    def __next__(self):
        if self.label is None:
            return torch.randn(self.shape, device=self.device)
        else:
            return (
                torch.randn(self.shape, device=self.device),
                torch.full((self.shape[0],), self.label, device=self.device),
            )


def get_dataloader(dataset_name: str, fixed_label: int | None = None) -> DataLoader:
    """ """
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
        target_transform=lambda x: torch.nn.functional.one_hot(
            torch.Tensor([x]).long(), output_dim
        )
        .float()
        .squeeze(),
    )
    if fixed_label is not None:
        dataset = CustomDataset(dataset, fixed_label)

    return DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
    )
