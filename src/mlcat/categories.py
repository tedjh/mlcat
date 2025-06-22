"""Classes relating to Categories"""

from dataclasses import dataclass
from typing import OrderedDict
import torch

from mlcat.objects import Object
from mlcat.morphisms import TorchMorphism, AbstractMorphism, ProductObject


class CategoryError(Exception):
    """Exception raised if an error occurs in :class:`Category`"""


class PathError(Exception):
    """PathError"""


class FunctorError(Exception):
    """Exception raised if an error occurs in :class:`Functor`"""


@dataclass
class Path:
    """A Path is a sequence of Morphisms from one Object to another."""

    path: list[AbstractMorphism]
    optimiser: torch.optim.Optimizer | None

    def __post_init__(self):
        if not self.check_is_path(self.path):
            raise PathError("list of Morphisms is not a valid path")

        self.dom = self.path[0].dom
        self.codom = self.path[-1].codom

        torch_morphisms = sorted(
            [i for i, mor in enumerate(self.path) if isinstance(mor, TorchMorphism)]
        )
        if torch_morphisms:
            if any(
                i not in torch_morphisms
                for i in range(min(torch_morphisms), max(torch_morphisms))
            ):
                raise PathError(
                    "Not implemented non-torch Modules between torch Modules"
                )

            torch_path_morphism = TorchMorphism(
                self.path[min(torch_morphisms)].dom,
                self.path[max(torch_morphisms)].codom,
                func=torch.nn.Sequential(
                    OrderedDict(
                        {
                            str(i): morphism.func
                            for i, morphism in enumerate(
                                self.path[
                                    min(torch_morphisms) : max(torch_morphisms) + 1
                                ]
                            )
                        }
                    )
                ),
            )
            self.path = [
                *self.path[: min(torch_morphisms)],
                torch_path_morphism,
                *self.path[max(torch_morphisms) + 1 :],
            ]

        def path_forward(x) -> torch.Tensor:
            for mor in self.path:
                x = mor(x)
            return x

        self.func = path_forward

    @staticmethod
    def check_is_path(morphisms: list[AbstractMorphism]) -> bool:
        """Check if a list of morphisms is a valid path, i.e. each morphism's codomain
        is the same as the next morphism's domain."""
        for mor_a, mor_b in zip(morphisms, morphisms[1:]):
            if mor_a.codom != mor_b.dom:
                return False

        return True


@dataclass
class Equation:
    """An Equation is a pair of Paths whose equality is what will be optimised for,
    with respect to a given dataloader and loss function to use for training."""

    path_1: Path
    path_2: Path
    dataloader_name: str
    loss_function_name: str

    def __post_init__(self):
        if (self.path_1.dom != self.path_2.dom) or (
            self.path_1.codom != self.path_2.codom
        ):
            raise CategoryError("Paths in Equation have different (co)domains")
        if self.dataloader_name not in self.path_1.dom.dataloaders.keys():
            raise CategoryError("Dataloader not found in domain Object of path")
        if self.path_1.codom.loss_functions is None or (
            self.loss_function_name not in self.path_1.codom.loss_functions.keys()
        ):
            raise CategoryError("Loss function not found in codomain Object of path")


class Category:
    """A Category is a collection of Objects and Morphisms, with a set of Equations
    that define relationships between the Morphisms. It can be used to train models"""

    def __init__(
        self,
        morphisms: list[AbstractMorphism],
        device: torch.device,
        equations: list[Equation] | None = None,
    ) -> None:
        self.objects: list[Object | ProductObject] = []
        self.morphisms: list[AbstractMorphism] = []
        self.equations: list[Equation] = []
        self.unique_optimisers: list[tuple[int, int]] = []
        self.device = device
        for morphism in morphisms:
            if morphism.dom not in self.objects:
                self.objects.append(morphism.dom)
            if morphism.codom not in self.objects:
                self.objects.append(morphism.codom)
            if isinstance(morphism, TorchMorphism):
                morphism.add_device(device)
            self.morphisms.append(morphism)

        if equations is not None:
            for equation in equations:
                self.add_equation(equation)

    def add_equation(self, equation: Equation) -> None:
        """Add an Equation to the Category, checking that it is valid."""
        self.equations.append(equation)
        self.update_unique_optimisers(self.equations[-1])

    def update_unique_optimisers(self, equation) -> None:
        """Update the list of unique optimisers in the Category based on the given
        Equation."""
        if equation.path_1.optimiser is not None:
            self._update_unique_optimisers(equation.path_1.optimiser, path_1=True)
        if equation.path_2.optimiser is not None:
            self._update_unique_optimisers(equation.path_2.optimiser, path_1=False)

    def _update_unique_optimisers(
        self, optimiser: torch.optim.Optimizer, path_1: bool
    ) -> None:
        """
        Internal method to update the unique optimisers based on the given Equation.
        """
        optim_1_params = set(
            id(p) for group in optimiser.param_groups for p in group["params"]
        )
        for equation_num, path_num in self.unique_optimisers:
            eq = self.equations[equation_num]
            path = eq.path_1 if path_num == 0 else eq.path_2
            assert path.optimiser is not None
            if (
                set(
                    id(p)
                    for group in path.optimiser.param_groups
                    for p in group["params"]
                )
                == optim_1_params
            ):
                break
        else:
            self.unique_optimisers.append((len(self.equations) - 1, 0 if path_1 else 1))

    def train(self, total_epochs: int) -> None:
        """Train the Category for a given number of epochs."""
        unique_dataloaders: dict[tuple[ProductObject, str], list[int]] = {}
        for i, equation in enumerate(self.equations):
            if (
                loader_details := (
                    equation.path_1.dom,
                    equation.dataloader_name,
                )
            ) not in unique_dataloaders:
                unique_dataloaders[loader_details] = [i]
            else:
                unique_dataloaders[loader_details].append(i)

        if any(
            sum(i in v for v in unique_dataloaders.values()) > 1
            for i in range(len(self.equations))
        ):
            raise CategoryError("Equation was deemed to have two unique dataloaders")

        self._train(unique_dataloaders, total_epochs)

    def _train(
        self,
        unique_dataloaders: dict[tuple[ProductObject, str], list[int]],
        total_epochs: int,
    ) -> None:
        """Internal method to train the Category, given the unique dataloaders"""
        print(f"Training on device {self.device}...")
        for epoch in range(total_epochs):
            print(f"Epoch: {epoch}")
            for tensor_tuple in zip(
                *[
                    data_space_object.dataloaders[dataloader_name]
                    for data_space_object, dataloader_name in unique_dataloaders
                ]
            ):
                equation_losses: dict[int, torch.Tensor] = {}
                for j, image_label_tuple in enumerate(tensor_tuple):
                    image_label_tuple = (
                        image_label_tuple[0].to(device=self.device),
                        image_label_tuple[1].to(device=self.device),
                    )
                    for i in list(unique_dataloaders.values())[j]:
                        eq = self.equations[i]
                        if (
                            eq.path_1.codom.loss_functions is None
                            or (
                                loss_fn := eq.path_1.codom.loss_functions.get(
                                    eq.loss_function_name
                                )
                            )
                            is None
                        ):
                            raise CategoryError(
                                f"Loss function {eq.loss_function_name} not found in "
                                f"codomain of path {eq.path_1}"
                            )
                        equation_losses[i] = loss_fn(
                            eq.path_1.func(image_label_tuple),
                            eq.path_2.func(image_label_tuple),
                        )

                for i, equation in enumerate(self.equations):
                    if equation.path_1.optimiser is not None:
                        equation.path_1.optimiser.zero_grad()
                    if equation.path_2.optimiser is not None:
                        equation.path_2.optimiser.zero_grad()

                    equation_losses[i].backward()

                    if equation.path_1.optimiser is not None:
                        equation.path_1.optimiser.step()
                    if equation.path_2.optimiser is not None:
                        equation.path_2.optimiser.step()
