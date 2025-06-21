"""Classes relating to Categories"""

import torch
from typing import OrderedDict
from dataclasses import dataclass

from mlcat.objects import Object
from mlcat.morphisms import Projection, TorchMorphism, AbstractMorphism, ProductObject


class CategoryError(Exception):
    """Exception raised if an error occurs in :class:`Category`"""


class TrainPathError(Exception):
    """TrainPathError"""


class FunctorError(Exception):
    """Exception raised if an error occurs in :class:`Functor`"""


@dataclass
class TrainPath:
    path: list[Projection | TorchMorphism]
    optimiser: torch.optim.Optimizer | None

    def __post_init__(self):
        if not self.check_is_path(self.path):
            raise TrainPathError("list of Morphisms is not a valid path")

        torch_morphisms = sorted(
            [i for i, mor in enumerate(self.path) if isinstance(mor, TorchMorphism)]
        )
        if torch_morphisms:
            if any(
                [
                    i not in torch_morphisms
                    for i in range(min(torch_morphisms), max(torch_morphisms))
                ]
            ):
                raise TrainPathError(
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
            self.reduced_path = [
                *self.path[: min(torch_morphisms)],
                torch_path_morphism,
                *self.path[max(torch_morphisms) + 1 :],
            ]

            def path_forward(x) -> torch.Tensor:
                for mor in self.reduced_path:
                    x = mor(x)
                return x

        else:

            def path_forward(x) -> torch.Tensor:
                for mor in self.path:
                    x = mor(x)
                return x

        self.path_morphism = AbstractMorphism(
            self.path[0].dom,
            self.path[-1].codom,
            path_forward,
        )

    @staticmethod
    def check_is_path(morphisms: list[AbstractMorphism]) -> bool:
        for mor_a, mor_b in zip(morphisms, morphisms[1:]):
            if mor_a.codom != mor_b.dom:
                return False

        return True


@dataclass
class Equation:
    path_1: TrainPath
    path_2: TrainPath
    dataloader_name: str
    loss_function_name: str

    def __post_init__(self):
        if (self.path_1.path_morphism.dom != self.path_2.path_morphism.dom) or (
            self.path_1.path_morphism.codom != self.path_2.path_morphism.codom
        ):
            raise CategoryError("Paths in Equation have different (co)domains")
        if self.dataloader_name not in self.path_1.path_morphism.dom.dataloaders.keys():
            raise CategoryError("Dataloader not found in domain Object of path")
        if (
            self.loss_function_name
            not in self.path_1.path_morphism.codom.loss_functions.keys()
        ):
            raise CategoryError("Loss function not found in codomain Object of path")


class Category:
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
        self.equations.append(equation)
        self.update_unique_optimisers(self.equations[-1])

    def update_unique_optimisers(self, equation) -> None:
        if equation.path_1.optimiser is not None:
            optim_1_params = set(
                id(p)
                for group in equation.path_1.optimiser.param_groups
                for p in group["params"]
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
                self.unique_optimisers.append((len(self.equations) - 1, 0))

        if equation.path_2.optimiser is not None:
            optim_2_params = set(
                id(p)
                for group in equation.path_2.optimiser.param_groups
                for p in group["params"]
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
                    == optim_2_params
                ):
                    break
            else:
                self.unique_optimisers.append((len(self.equations) - 1, 1))

    def train(self, total_epochs: int) -> None:
        unique_dataloaders: dict[tuple[ProductObject, str], list[int]] = {}
        for i, equation in enumerate(self.equations):
            if (
                loader_details := (
                    equation.path_1.path_morphism.dom,
                    equation.dataloader_name,
                )
            ) not in unique_dataloaders.keys():
                unique_dataloaders[loader_details] = [i]
            else:
                unique_dataloaders[loader_details].append(i)

        if any(
            sum(i in v for _, v in unique_dataloaders.items()) > 1
            for i in range(len(self.equations))
        ):
            raise CategoryError("Equation was deemed to have two unique dataloaders")

        print(f"Training on device {self.device}...")
        for epoch in range(total_epochs):
            print(f"Epoch: {epoch}")
            for tensor_tuple in zip(
                *[
                    data_space_object.dataloaders[dataloader_name]
                    for data_space_object, dataloader_name in unique_dataloaders.keys()
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
                        equation_losses[
                            i
                        ] = eq.path_1.path_morphism.codom.loss_functions[
                            eq.loss_function_name
                        ](
                            eq.path_1.path_morphism(image_label_tuple),
                            eq.path_2.path_morphism(image_label_tuple),
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
