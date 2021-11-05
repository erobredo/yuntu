"""Acoustic indices."""
from abc import ABC
from abc import abstractmethod

class AcousticIndex(ABC):
    """Base class for acoustic indices."""
    name = None
    multi = False
    ncomponents = 1

    def __init__(self, name=None):
        if name is not None:
            self.name = name

    def __call__(self, array):
        """Call method for class."""
        return self.run(array)

    @abstractmethod
    def run(self, array):
        """Run transformations and return index."""
