"""Metaclass of model.

This class indicates a list of methods
to be implemented in concrete model classes.
"""

from abc import ABCMeta, abstractmethod


class Model(metaclass=ABCMeta):
    """Meta class of model."""

    def __init__(self, name):
        """Initialize Model."""
        self.name = name

    @abstractmethod
    def compile(self, input_shape, output_shape):
        """Configure model for training."""
        pass

    def set_extra_config(self, *kwargs):
        """Set extra config after generating this object.

        :param kwargs:
        """
        pass
