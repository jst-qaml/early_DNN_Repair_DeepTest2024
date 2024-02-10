"""Metaclass of dataset.

This class indicates a list of methods
to be implemented in concrete dataset classes.
"""

import importlib
import os
from abc import ABCMeta, abstractmethod
from pathlib import Path
import csv

import numpy as np
import random

import h5py

from . import test, train, eAIDataset


class Dataset(metaclass=ABCMeta):
    """Meta class of dataset."""

    def __init__(self, name):
        """Initialize Dataset."""
        self.name = name

    @abstractmethod
    def _get_input_shape(self):
        """Return the input_shape and classes in this function.

        return: two values as below
            1.A three-values tuple like (height, width, color) and
            2.The number of classes as the output result

        """
        pass

    @abstractmethod
    def prepare(self, input_dir, output_dir, divide_rate, random_state):
        """Prepare dataset to train.

        :param input_dir:
        :param output_dir:
        :param divide_rate:
        :param random_state:

        """
        pass

    def train(self,
              model,
              batch_size=32,
              epochs=50,
              validation_split=0.2,
              gpu=False,
              data_dir=r'outputs/',
              output_dir=r'outputs/',
              weighted_classes=False):
        """Train.

        :param model:
        :param batch_size:
        :param epochs:
        :param validation_split:
        :param gpu:
        :param data_dir:
        :param output_dir:
        :param weighted_classes:
        :return:

        """
        output_dir = Path(output_dir)
        input_shape, classes = self._get_input_shape()

        os.environ['H5PY_DEFAULT_READONLY'] = '1'
        train.train(model,
                    input_shape,
                    classes,
                    batch_size,
                    epochs,
                    validation_split,
                    gpu,
                    data_dir,
                    output_dir,
                    weighted_classes)

        return
    
    def finetune(self,
            model_dir,
            seed,
            batch_size=32,
            epochs=50,
            validation_split=0.2,
            gpu=False,
            data_dir=r'outputs/',
            output_dir=r'outputs/',
            weighted_classes=False,
            class_weights=None):
        """Train.

        :param model:
        :param batch_size:
        :param epochs:
        :param validation_split:
        :param gpu:
        :param data_dir:
        :param output_dir:
        :param weighted_classes:
        :return:

        """
        output_dir = Path(output_dir)
        input_shape, classes = self._get_input_shape()

        os.environ['H5PY_DEFAULT_READONLY'] = '1'
        train.finetune(model_dir,
                    seed,
                    input_shape,
                    classes,
                    batch_size,
                    epochs,
                    validation_split,
                    gpu,
                    data_dir,
                    output_dir,
                    weighted_classes,
                    class_weights)

        return

    def test(self, model_dir, data_dir, target_data, batch_size, print_pos_neg_indexes=False):
        """Test.

        :param model_dir:
        :param data_dir:
        :param target_data:
        :param batch_size:
        :param print_pos_neg_indexes:
        :return:
        """
        model_dir = Path(model_dir)
        data_dir = Path(data_dir)

        test.test(model_dir, data_dir, target_data, batch_size, print_pos_neg_indexes)

        return

    def target(self, model_dir, data_dir, batch_size, dataset_type='repair', do_cleanup=True):
        """Find target dataset.

        :param model_dir:
        :param data_dir:
        :param batch_size:
        :param dataset_type:
        :param do_cleanup:
        :return:
        """
        model_dir = Path(model_dir)
        data_dir = Path(data_dir)

        test.target(model_dir, data_dir, batch_size, dataset_type, do_cleanup)

        return

    def load_test_data(self, data_dir):
        """Load test data.

        :param data_dir:
        :return:
        """
        return self._load_data(data_dir, r'test.h5')

    def load_repair_data(self, data_dir):
        """Load repair data.

        :param data_dir:
        :return:
        """
        return self._load_data(data_dir, r'repair.h5')

    def load_repair_data_split(self, data_dir):
        """Load repair data. Unlike self.load_repair_data(), this
        method loads repair data and splits it according to how it is
        misclassified.
        :param data_dir:
        :return:
        """
        data_dir = Path(data_dir)
        repair_images = {}
        repair_labels = {}

        for subclass in os.scandir(data_dir):
            if subclass.is_dir() and subclass.name in [str(x) for x in self.get_classes()]:
                class_name = subclass.name
                repair_images[class_name], repair_labels[class_name] = self._load_data(subclass.path,r'repair.h5')

        return repair_images, repair_labels

    def _load_data(self, data_dir, target):
        """Load data from the input file.

        :param data_dir:
        :param target:
        :return:
        """
        data_dir = str(data_dir)
        target + str(target)
        if data_dir[-1] is not "/" and target[0] is not "/":
            target = "/"+target
        dataset = eAIDataset.EAIDataset(data_dir + target)
        return dataset

    def load_train_model(self, model_dir):
        """Load model.

        :param model_dir:
        :return:
        """
        return train.load_model(model_dir)


    def load_model(self, model_dir):
        """Load model.

        :param model_dir:
        :return:
        """
        return test.load_model(model_dir)

    def utils(self, kwargs):
        """Utilities.

        :param kwargs:
        """
        if 'call' in kwargs:
            try:
                call = kwargs['call']
                module = importlib.import_module('.' + call, 'dataset.utils')
                method = getattr(module, call)
            except Exception:
                raise Exception('Invalid call: {}'.format(call))
            method(self, kwargs)
        else:
            raise Exception('Require --call')

    def set_extra_config(self, *kwargs):
        """Set extra config after generating this object.

        :param kwargs:
        """
        pass

    def get_classes(self):

        classes_dict = {
            'GTSRB': list(range(0, 43)),
            'CIFAR_10': list(range(0, 10)),
            'BDD_OBJECTS': [0, 1, 2, 3, 6, 7, 8, 9, 12]
        }

        return classes_dict[self.name.upper()]

    def count_samples(self, dataset_type='repair', path_to_data=None):
        # Count the number of misclassified and correctly-classified images
        # for each type of misclassification. We do it once in the beginning
        # to avoid performance degradation
        misclassified = {}
        well_classified = {}

        classes = self.get_classes()

        # Initialize the counters to all 0
        for cl1 in classes:

            for cl2 in classes:
                mscl = (cl1, cl2)
                misclassified[mscl] = 0
                well_classified[mscl[0]] = 0

        # Navigate the subfolders of the outputs/negative folder
        # in order to count the total misclassified images (for each class)
        if path_to_data is None:
            output_path = Path('outputs').joinpath(self.name.upper().replace("_", "-")).joinpath('negative')
        else:
            output_path = Path(path_to_data).joinpath('negative/')

        for class_1_dir in os.scandir(output_path):

            if class_1_dir.is_dir() and class_1_dir.name in [str(x) for x in self.get_classes()]:

                print("Found misclassifications from class", class_1_dir.name, end=" to\n")

                for class_2_dir in os.scandir(class_1_dir):

                    if class_2_dir.is_dir() and class_2_dir.name in [str(x) for x in self.get_classes()]:
                        print("class", class_2_dir.name, end="")

                        misclassification = (int(class_1_dir.name), int(class_2_dir.name))

                        file_path = Path(class_2_dir.path).joinpath(f'{dataset_type}.h5')
                        if file_path.is_file():
                            data = h5py.File(file_path, 'r')
                            img_amount = len(data['labels'][()])
                            data.close()
                        else:
                            img_amount = 0

                        if misclassification in misclassified.keys():
                            misclassified[misclassification] += img_amount
                        else:
                            assert False
                            # misclassified[misclassification] = 0

                        print(":", misclassified[(int(class_1_dir.name), int(class_2_dir.name))])

        # Now navigate through the positive inputs and count
        # the correctly classified images
        if path_to_data is None:
            output_path = Path('outputs/').joinpath(self.name.upper().replace("_", "-")).joinpath('positive')
        else:
            output_path = Path(path_to_data).joinpath('positive/')

        for class_dir in os.scandir(output_path):

            if class_dir.is_dir():

                file_path = Path(class_dir.path).joinpath(f'{dataset_type}.h5')
                if file_path.is_file():
                    data = h5py.File(file_path, 'r')
                    img_amount = len(data['labels'][()])
                    data.close()
                else:
                    img_amount = 0

                if int(class_dir.name) in well_classified.keys():
                    well_classified[int(class_dir.name)] += img_amount
                else:
                    well_classified[int(class_dir.name)] = 0

        if path_to_data is None:
            return misclassified, well_classified
        else:
            with open(f'{path_to_data}{dataset_type}_misclassifNumbers.csv', 'w') as csvfile:
                writer = csv.DictWriter(csvfile, misclassified.keys())
                writer.writeheader()
                writer.writerow(misclassified)
            with open(f'{path_to_data}{dataset_type}_correctNumbers.csv', 'w') as csvfile:
                writer = csv.DictWriter(csvfile, well_classified.keys())
                writer.writeheader()
                writer.writerow(well_classified)

