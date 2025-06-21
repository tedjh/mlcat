"""Classes relating to the Objects of Categories"""

from typing import Callable
import torch


class ObjectError(Exception):
    """Exception raised if an error occurs in :class:`Object`"""


class Object:
    def __init__(
        self,
        shape: torch.Size,
        loss_functions: (
            dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] | None
        ) = None,
    ):
        self.shape = shape
        self.loss_functions = loss_functions

        if self.loss_functions is not None:
            for _, loss_function in self.loss_functions.items():
                try:
                    dist = loss_function(torch.ones(self.shape), torch.ones(self.shape))
                    dist.item()
                except Exception:
                    raise ObjectError(
                        "Expected loss function input does not match object shape"
                    )

    def __repr__(self):
        return f"{type(self).__name__}(Shape:{tuple(self.shape)})"

    def __eq__(self, other):
        if type(self).__name__ != type(other).__name__:
            return False
            # msg = (
            #    f"Cannot compare object of type {type(self).__name__} with object "
            #    f"of type {type(other).__name__}"
            # )
            # raise ObjectError(msg)

        return self.shape == other.shape

    def __hash__(self):
        return hash(tuple(self.shape))

    def check_is_in(self, x: torch.Tensor) -> bool:
        if x.ndim == len(self.shape):
            return x.shape == self.shape
        return x[0].shape == self.shape
