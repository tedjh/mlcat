import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Object:
    name: str


@dataclass(frozen=True)
class Morphism:
    name: str
    domain: Object
    codomain: Object


class IdentityMorphism(Morphism):
    def __init__(self, name: str, domain: Object):
        super().__init__(name, domain, domain)


class Category:
    def __init__(self, objects: list[Object], morphisms: list[Morphism]):
        if len(objects) != len(set([cat_obj.name for cat_obj in objects])):
            raise Exception("The given objects do not have unique names")
        if len(morphisms) != len(set([cat_morphism.name for cat_morphism in morphisms])):
            raise Exception("The given morphisms do not have unique names")

        self.objects: set[Object] = set(objects)
        self.morphisms: set[Morphism] = set(morphisms)

        # Check (and correct, if possible) the objects and morphisms are valid to make a category
        self.check_consistency()

    def check_consistency(self):
        """Checks various basic properties. Does not check that all morphisms that could exist are defined, as
        there can be infinitely many"""
        # Check the (co)domains of all morphisms are objects in the category
        for morphism in self.morphisms:
            if not (morphism.domain in self.objects and morphism.codomain in self.objects):
                raise Exception("Morphism's domain/codomain not on object in this category.")

        # Check each object has precisely one identity morphism
        for cat_object in self.objects:
            # For each object, determine how many IdentityMorphisms are defined over it.
            num_identity_morphisms = len([morphism for morphism in self.morphisms if (
                    isinstance(morphism, IdentityMorphism) and morphism.domain is cat_object
            )])
            # If none are defined, add one
            if num_identity_morphisms == 0:
                self.morphisms.add(IdentityMorphism(f"id_{cat_object.name}", cat_object))
            # If more than one exists, raise Exception.
            elif num_identity_morphisms > 1:
                raise Exception(f"Object {cat_object.name} should have precisely one identity morphism.")

    def show(self):
        cat_graph = nx.DiGraph()
        cat_graph.add_nodes_from(self.objects)
        cat_graph.add_edges_from([(morphism.domain, morphism.codomain) for morphism in self.morphisms])
        nx.draw(cat_graph)
        plt.draw()
        plt.show()

    def compose(self, f: Morphism, g: Morphism):
        """Attempts to make the composed morphism `g \circ f`.
        Adds to the category's morphism set if not already there"""
        if not (f in self.morphisms and g in self.morphisms):
            raise Exception("Morphisms must be within the same category")

        if g.domain != f.codomain:
            raise Exception(f"The morphism {g.name} cannot be composed with {f.name}")

        if isinstance(f, IdentityMorphism):
            return g
        elif isinstance(g, IdentityMorphism):
            return f
        else:
            composed_morphism = Morphism(f"{g.name}_{f.name}", f.domain, g.codomain)
            # If g\circ f isn't present in category's morphism set already then add it.
            if composed_morphism not in self.morphisms:
                self.morphisms.add(composed_morphism)
                # As a new morphism has been added, we should check consistency again
                self.check_consistency()

            return composed_morphism


class Functor:
    def __init__(
            self,
            name: str,
            domain: Category,
            codomain: Category,
            object_map: Callable[[Object], Object],
            morphism_map: Callable[[Morphism], Morphism]
    ):
        self.name = name
        self.domain = domain
        self.codomain = codomain
        self.object_map = object_map
        self.morphism_map = morphism_map

    def __call__(self, object: Object | None = None, morphism: Morphism | None = None):
        if object is None and morphism is None:
            raise Exception("Functor must act on something that isn't None")
        elif isinstance(object, Object) and isinstance(morphism, Morphism):
            raise Exception("Only one of object and morphism can be given at a time")
        elif isinstance(object, Object):
            mapped_object = self.object_map(object)
            if mapped_object not in self.codomain.objects:
                raise Exception("Mapped object is not inside the codomain category")
            return mapped_object
        elif isinstance(morphism, Morphism):
            mapped_morphism = self.morphism_map(morphism)
            if mapped_morphism not in self.codomain.morphisms:
                raise Exception("Mapped morphism not found inside the codomain category")
            return mapped_morphism
        else:
            raise Exception("Unknown situation occurred")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    A, B, C = Object("A"), Object("B"), Object("C")
    f, g = Morphism("f", A, B), Morphism("g", B, C)
    my_cat = Category(
        [A, B, C],
        [f, g]
    )
    my_cat.show()
    h = my_cat.compose(f, g)
    my_cat.show()

