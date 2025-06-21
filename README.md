# Plan

## Setup
1. Create a uv environment via:
   1. `uv venv`
   2. `.venv\Scripts\activate` (on Windows)
2. Install packages via: `uv pip install .[dev]`
3. Setup pre-commit: `pre-commit install`


# Overview of

- Objects represent tensor spaces, and so are defined by a torch.Size object. They can
optionally have metrics defined on them, which form loss functions for models whose
outputs are elements of this object. For instance, `torch.nn.functional.cross_entropy`
can be attached to the object defined by `torch.Size([10])`, for instance.
- Morphisms will be either:
  - regular functions (i.e. un-parameterised), like a custom function or
    torch.nn.functional. In this case we should manually define the source and target
    objects.
  - torch.nn.Module (parameterised), we can determine the source and target objects if
    this Module has `in/out_features` attributes. This kind of morphism will be
    optimised if it is part of an Equation.
- Categories will be defined via a set of generating morphisms. The objects of the
category will be determined by the source and targets of the generating morphisms.
