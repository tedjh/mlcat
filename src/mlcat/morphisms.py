"""Classes relating to the Morphisms of Categories"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar, Callable, Iterable, Any
import torch

from mlcat.objects import Object, ObjectError
from mlcat.data import RandomImageGenerator


class MorphismError(Exception):
    """Exception raised if an error occurs in :class:`Morphism`"""


@dataclass
class ProductObject:
    """A ProductObject is a tuple of two Objects"""

    a: Object
    b: Object
    dataloaders: dict[
        str, Iterable[tuple[torch.Tensor, torch.Tensor]] | RandomImageGenerator
    ]

    def __post_init__(self):
        self.projection_a = Projection(self, self.a)
        self.projection_b = Projection(self, self.b)

        for dataloader_name, dataloader in self.dataloaders.items():
            test_tensor, test_label = next(iter(dataloader))
            # Note it is always assumed the first index is the batch index, which has
            # no restrictions.
            if not (
                self.a.check_is_in(test_tensor[0]) and self.b.check_is_in(test_label[0])
            ):
                raise ObjectError(
                    f"Dataloader {dataloader_name} does not produce data of correct "
                    f"shape for objects {self.a} and {self.b}. "
                    f"Got: a={test_tensor.shape}, b={test_label.shape}"
                )


D = TypeVar("D")
F = TypeVar("F")


class AbstractMorphism(ABC, Generic[D, F]):
    """An abstract class representing a morphism between two objects in a category"""

    def __init__(
        self,
        dom: D,
        codom: Object,
        func: F,
    ):
        self.dom = dom
        self.codom = codom
        self.func = func

    @abstractmethod
    def __call__(self, x) -> torch.Tensor:
        """Forward pass of the morphism, applying the function to the input x"""

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        """Check if two morphisms are equal, i.e. have the same domain, codomain and
        function"""


class TorchMorphism(AbstractMorphism[Object, torch.nn.Module]):
    """A morphism whose function is a torch Module"""

    def __init__(
        self,
        dom: Object,
        codom: Object,
        func: torch.nn.Module,
    ):
        super().__init__(dom, codom, func)

    def add_device(self, device: torch.device) -> None:
        """Move the function to the specified device"""
        assert isinstance(self.func, torch.nn.Module)
        self.func.to(device=device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.func(x)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, AbstractMorphism):
            raise MorphismError("Cannot compare morphism with non-morphism")
        if self.dom != other.dom or self.codom != other.codom:
            return False
        if not isinstance(other.func, torch.nn.Module):
            return False
        if list(self.func.named_children()) != list(other.func.named_children()):
            return False
        if (sd := self.func.state_dict()).keys() != other.func.state_dict().keys():
            return False
        return all(
            torch.equal(sd_value, other.func.state_dict()[sd_key])
            for sd_key, sd_value in sd.items()
        )


class Projection(
    AbstractMorphism[
        ProductObject, Callable[[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]
    ]
):
    """A morphism that projects onto one of the legs of a ProductObject"""

    def __init__(
        self,
        dom: ProductObject,
        codom: Object,
    ):
        if codom not in [dom.a, dom.b]:
            raise MorphismError(
                f"Codomain {codom} is not equal to either leg of dom: {dom.a}, "
                f"{dom.b}"
            )

        def proj(x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
            return x[0] if dom.a == codom else x[1]

        super().__init__(dom, codom, proj)

    def __call__(self, x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self.func(x)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, AbstractMorphism):
            raise MorphismError("Cannot compare morphism with non-morphism")
        if self.dom != other.dom or self.codom != other.codom:
            return False
        return True
