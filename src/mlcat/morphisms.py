"""Classes relating to the Morphisms of Categories"""

import torch
from torch.utils.data import DataLoader
from typing import Callable

from mlcat.objects import Object, ObjectError
from mlcat.data import RandomGenerator


class MorphismError(Exception):
    """Exception raised if an error occurs in :class:`Morphism`"""


class ProductObject:
    def __init__(
        self,
        a: Object,
        b: Object,
        dataloaders: dict[str, DataLoader | RandomGenerator],
    ):
        self.a = a
        self.b = b
        self.dataloaders = dataloaders
        self.projection_a = Projection(self, self.a)
        self.projection_b = Projection(self, self.b)

        for dataloader_name, dataloader in self.dataloaders.items():
            test_tensor, test_label = next(iter(dataloader))
            # Note it is always assumed the first index is the batch index, which has
            # no restrictions.
            if not (a.check_is_in(test_tensor[0]) and b.check_is_in(test_label[0])):
                raise ObjectError(
                    f"Dataloader {dataloader_name} does not produce data of correct "
                    f"shape for objects {a} and {b}. "
                    f"Got: a={test_tensor.shape}, "
                    f"b={test_label.shape}"
                )


class AbstractMorphism:
    def __init__(
        self,
        dom: Object | ProductObject,
        codom: Object,
        func: (
            torch.nn.Module
            | Callable[[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]
        ),
    ):
        self.dom = dom
        self.codom = codom
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __eq__(self, other):
        if not isinstance(other, AbstractMorphism):
            raise MorphismError("Cannot compare morphism with non-morphism")
        if self.dom != other.dom or self.codom != other.codom:
            return False
        if not isinstance(self.func, type(other.func)):
            return False
        if isinstance(self.func, torch.nn.Module):
            if list(self.func.named_children()) != list(other.func.named_children()):
                return False
            if (sd := self.func.state_dict()).keys() != other.func.state_dict().keys():
                return False
            return all(
                torch.equal(sd_value, other.func.state_dict()[sd_key])
                for sd_key, sd_value in sd.items()
            )
        return True


class TorchMorphism(AbstractMorphism):
    def __init__(
        self,
        dom: Object,
        codom: Object,
        func: torch.nn.Module,
    ):
        """A morphism whose function is a torch Module"""
        super().__init__(dom, codom, func)

    def add_device(self, device: torch.device) -> None:
        assert isinstance(self.func, torch.nn.Module)
        self.func.to(device=device)


class Projection(AbstractMorphism):
    def __init__(
        self,
        dom: ProductObject,
        codom: Object,
    ):
        if dom.a != codom and dom.b != codom:
            raise MorphismError(
                f"Codomain {codom} is not equal to either leg of dom: {dom.a}, "
                f"{dom.b}"
            )

        def proj(x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
            return x[0] if dom.a == codom else x[1]

        super().__init__(dom, codom, proj)
