"""Generative Adversarial Network (GAN) example using the mlcat library."""

import torch

from mlcat.objects import Object
from mlcat.morphisms import TorchMorphism, ProductObject
from mlcat.categories import Category, Path, Equation
from mlcat.data import get_dataloader, RandomImageGenerator


def main(dataset_name: str = "MNIST"):
    """
    Main function to set up and train a GAN using the mlcat library.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    classification_space = Object(
        torch.Size([1]), {"BCE": torch.nn.functional.binary_cross_entropy}
    )
    image_space = Object(torch.Size([28 * 28]), None)

    train_space = ProductObject(
        image_space,
        classification_space,
        dataloaders={dataset_name: get_dataloader(dataset_name, fixed_label=1)},
    )

    discriminator = TorchMorphism(
        image_space,
        classification_space,
        torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid(),
        ),
    )

    latent_dim = 100
    latent_space = Object(torch.Size([latent_dim]), loss_functions=None)

    latent_train_space = ProductObject(
        latent_space,
        classification_space,
        dataloaders={
            "generated_real": RandomImageGenerator(
                shape=torch.Size([batch_size, latent_dim]), device=device, label=1
            ),
            "generated_fake": RandomImageGenerator(
                shape=torch.Size([batch_size, latent_dim]), device=device, label=0
            ),
        },
    )

    generator = TorchMorphism(
        latent_space,
        image_space,
        torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 128),
            torch.nn.ReLU(True),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(True),
            torch.nn.Linear(256, 28 * 28),
            torch.nn.Tanh(),  # output in [-1, 1]
        ),
    )
    assert isinstance(discriminator.func, torch.nn.Module)
    assert isinstance(generator.func, torch.nn.Module)

    category = Category(
        morphisms=[
            generator,
            train_space.projection_a,
            train_space.projection_b,
            discriminator,
            latent_train_space.projection_b,
            latent_train_space.projection_a,
        ],
        device=device,
        equations=[
            Equation(
                path_1=Path(
                    [train_space.projection_a, discriminator],
                    torch.optim.Adam(discriminator.func.parameters(), lr=0.001),
                ),
                path_2=Path([train_space.projection_b], None),
                dataloader_name="MNIST",
                loss_function_name="BCE",
            ),
            Equation(
                path_1=Path(
                    [latent_train_space.projection_a, generator, discriminator],
                    torch.optim.Adam(generator.func.parameters(), lr=0.001),
                ),
                path_2=Path([latent_train_space.projection_b], None),
                dataloader_name="generator_real",
                loss_function_name="BCE",
            ),
            Equation(
                path_1=Path(
                    [latent_train_space.projection_a, generator, discriminator],
                    torch.optim.Adam(generator.func.parameters(), lr=0.001),
                ),
                path_2=Path([latent_train_space.projection_b], None),
                dataloader_name="generator_fake",
                loss_function_name="BCE",
            ),
        ],
    )
    category.train(10)


if __name__ == "__main__":
    main()
