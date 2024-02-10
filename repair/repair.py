"""Repair."""

from abc import ABCMeta, abstractmethod
from pathlib import Path


class Repair(metaclass=ABCMeta):
    """Meta class of Repair method."""

    def __init__(self, name, model_dir, target_data_dir):
        """Initialize Repair."""
        self.name = name
        self.model_dir = Path(model_dir)
        self.target_data_dir = Path(target_data_dir)

    @abstractmethod
    def set_options(self, dataset, kwargs):
        """Set options.

        :param dataset:
        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def localize(self, model, input_neg, output_dir=r'outputs/', *args):
        """Localize neural weight candidates to repair.

        :param output_dir: path to output directory
        :param model: a DNN model to be repaired
        :param input_neg: a set of inputs that reveal the fault
        :param args:
        :return:
        """
        pass

    @abstractmethod
    def optimize(self,
                 model,
                 weights,
                 input_neg,
                 input_pos,
                 output_dir=r'outputs/',
                 risk_aware=False,
                 *args,):
        """Optimize neural weight candidates to repair.

        :param model:
        :param weights:
        :param input_neg:
        :param input_pos:
        :param output_dir:
        :param: risk_aware:
        :param: args:
        :return:
        """
        pass

    @abstractmethod
    def save_weights(self, weights, output_dir):
        """Save neural weight candidates.

        :param weights:
        :param output_dir:
        :return:
        """
        pass

    @abstractmethod
    def load_weights(self, model, output_dir):
        """Load neural weight candidates.

        :param model:
        :param output_dir:
        :return:
        """
        pass

    @abstractmethod
    def load_input_neg(self, dir):
        """Load negative inputs.

        :param dir:
        :return:
        """
        pass

    @abstractmethod
    def load_input_pos(self, dir):
        """Load positive inputs.

        :param dir:
        :return:
        """
        pass

    @abstractmethod
    def evaluate(self,
                 dataset,
                 method,
                 model_dir,
                 target_data,
                 target_data_dir,
                 positive_inputs,
                 positive_inputs_dir,
                 output_dir,
                 num_runs):
        """Evaluate repairing performance.

        :param dataset:
        :param model_dir:
        :param target_data:
        :param target_data_dir:
        :param positive_inputs:
        :param positive_inputs_dir:
        :param output_dir:
        :param num_runs:
        :return:
        """
        pass
