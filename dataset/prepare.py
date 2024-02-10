"""Utils for dataset.

This module provides common functions for the prepare function.
"""
import random

import h5py


def divide_train(train_images, train_labels, divide_rate, random_state):
    """Divide the train data into normal training data and repair data.

    :train_image The list of original train images
    :train_label The list of original train labels
               (The order must be same as train_image)
    :divide_rate The rate of repair data (default:0.2)
    :random_state The seed data for sampling
    :return two image datasets(train, repair) and
            two label datasets(train, repair)
    """
    if random_state is not None:
        random.seed(int(random_state))

    sample_list = random.sample(range(len(train_images)),
                                int(len(train_images) * divide_rate))
    sample_list.sort(reverse=True)
    repair_images = []
    repair_labels = []
    for i in sample_list:
        repair_images.append(train_images.pop(i))
        repair_labels.append(train_labels.pop(i))

    return train_images, train_labels, repair_images, repair_labels


def save_prepared_data(images,
                       labels,
                       repair_images,
                       repair_labels,
                       test_images,
                       test_labels,
                       output_path):
    """Save prepared data as h5 files.

    :param images: Images for training,
    :param labels: Labels for training,
    :param repair_images: Images for repairing,
    :param repair_labels: Labels for repairing,
    :param test_images: Images for testing,
    :param test_labels: Labels for testing,
    :param output_path: The path to save the datasets
    :return:
    """
    with h5py.File(output_path.joinpath(r'train.h5'), 'w') as hf:
        hf.create_dataset('images', data=images)
        hf.create_dataset('labels', data=labels)
    print('train_images: {}'.format(images.shape))
    print('train_labels: {}'.format(labels.shape))

    with h5py.File(output_path.joinpath(r'repair.h5'), 'w') as hf:
        hf.create_dataset('images', data=repair_images)
        hf.create_dataset('labels', data=repair_labels)
    print('repair_images: {}'.format(repair_images.shape))
    print('repair_labels: {}'.format(repair_labels.shape))

    with h5py.File(output_path.joinpath(r'test.h5'), 'w') as hf:
        hf.create_dataset('images', data=test_images)
        hf.create_dataset('labels', data=test_labels)
    print('test_images: {}'.format(test_images.shape))
    print('test_labels: {}'.format(test_labels.shape))
