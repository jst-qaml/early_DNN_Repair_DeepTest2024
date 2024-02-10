"""Modules for testing and targeting the DNN model."""
import collections
import csv
import importlib
import json
import shutil
import sys
from pathlib import Path
from os import makedirs, path

import h5py
import numpy as np
from tensorflow import keras
from keras.models import model_from_json

import tensorflow as tf
from dataset import eAIDataset


def test(model_dir=r'outputs/',
         data_dir=r'outputs/',
         target_data=r'test.h5',
         batch_size=32,
         print_pos_neg_indexes=False):
    """Test.

    :param model_dir:
    :param data_dir:
    :param target_data:
    :param batch_size:
    :param print_pos_neg_indexes: if true, prints two lists of integers. The indices of the inputs correctly and
            incorrectly classified.
    :return:
    """
    # Load DNN model
    model = load_model(model_dir)
    # Load test images and labels
    dataset = _load_dataset(data_dir, target_data)
    images_generator, labels_generator = dataset.get_generators()

    pos_indexes = set()
    neg_indexes = set()

    # Obtain accuracy as evaluation result of DNN model with test dataset
    predictions = model.predict(images_generator, verbose=0)
    score_per_label = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    correct_amount = 0
    index = 0
    for lbl_batch in labels_generator:
        for lbl in lbl_batch:
            lbl = int(np.argmax(lbl))
            prediction = int(np.argmax(predictions[index]))
            real_file_index = dataset.get_real_index(index)
            if prediction == lbl:
                correct_amount += 1
                pos_indexes.add(real_file_index)
            else:
                neg_indexes.add(real_file_index)
            score_per_label[lbl][prediction] += 1
            score_per_label[lbl]["total"] += 1
            score_per_label["total"][prediction] += 1
            score_per_label["total"]["total"] += 1

            index += 1

    score = correct_amount / dataset.amount_of_images()  # divide by the total number of images
    print(f"tested dataset {target_data} on model {path.basename(model_dir)}")
    print_test_scores(score, score_per_label, (pos_indexes, neg_indexes, print_pos_neg_indexes, model_dir, data_dir))
    sys.stdout.flush()

    return score


def print_test_scores(score, score_per_label, indexes):
    print("general accuracy: %.5f%%" % (score * 100))
    print("precision and recall:")
    sorted_labels = list(score_per_label.keys())
    sorted_labels.remove("total")
    sorted_labels = sorted(sorted_labels)  # only so the output is more readable
    print("label \t precision \t recall")
    for lbl in sorted_labels:
        if score_per_label["total"][lbl] == 0:  # we never predicted that class
            print(str(lbl) + " 0-never-predicted" + " 0-never-predicted")
        else:
            print(str(lbl) + " \t" +
                  '%.5f' % (score_per_label[lbl][lbl] / (score_per_label["total"][lbl])) + "\t " +  # acc
                  '%.5f' % (score_per_label[lbl][lbl] / (score_per_label[lbl]["total"])))  # recall

    print("\ntotal images: " + str(score_per_label["total"]["total"]))
    print("per class:")
    print("label \t true_amount \t predicted_amount")
    for lbl in sorted_labels:
        print(str(lbl) + "\t" + str(score_per_label[lbl]["total"]) + " \t" + str(score_per_label["total"][lbl]))

    print("\ndetailed predictions:")
    print("true_label \t predicted \t amount")
    for true in sorted_labels:
        sorted_predictions = list(score_per_label[true].keys())
        sorted_predictions.remove("total")
        sorted_predictions = sorted(sorted_predictions)
        for predict in sorted_predictions:
            print(str(true) + "\t" + str(predict) + "\t" + str(score_per_label[true][predict]))

    pos_indexes, neg_indexes, print_pos_neg_indexes, model_dir, data_dir = indexes
    if print_pos_neg_indexes:
        log_path = str(model_dir) + "/pos_neg_indexes/"
        if not path.exists(log_path):
            makedirs(log_path)
        with open(f'{log_path}index_pos.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            pos_indexes = sorted(list(pos_indexes))
            writer.writerow(pos_indexes)
        with open(f'{log_path}index_neg.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            neg_indexes = sorted(list(neg_indexes))
            writer.writerow(neg_indexes)


def target(model_dir=r'outputs/', data_dir=r'outputs/', batch_size=32, dataset_type='repair', do_cleanup=True):
    """Find target dataset.

    :param model_dir:
    :param data_dir:
    :param batch_size:
    :param dataset_type: distinguishes between repair, train and test .h5 to separate in classes
    :param do_cleanup: determines if we should erase previous target results
    :return:
    """
    data_dir = Path(data_dir)
    print()
    # Load DNN model
    model = load_model(model_dir)
    # Load test images and labels
    datasets = _load_dataset(data_dir, r'repair.h5')
    test_images, test_labels = datasets.get_generators()

    # Predict labels from test images
    results = model.predict(test_images, verbose=0, batch_size=batch_size)

    # Parse and save predict/test results
    print("parse test")
    test_images, test_labels = datasets.get_generators()
    successes, failures = _parse_test_results(test_images,
                                              test_labels,
                                              results)
    save_path = str(model_dir) + '/' #path.basename(model_dir) + '/'
    print(f"save positive in model file {save_path}")
    _save_positive_results(successes, data_dir, save_path + r'positive/', dataset_type, do_cleanup)
    _save_negative_results(failures, data_dir, save_path + r'negative/', dataset_type, do_cleanup)

    _save_label_data(successes, data_dir.joinpath(save_path + r'positive/labels.json'))
    _save_label_data(failures, data_dir.joinpath(save_path + r'negative/labels.json'))

    return


def _parse_test_results(test_images, test_labels, results):
    """Parse test results.

    Parse test results and split them into success and failure datasets.
    Both datasets are dict of list consisted of dict of image and label.
    successes: {0: [{'image': test_image, 'label': test_label}, ...],
                1: ...}
    failures: {0: {1: [{'image': test_image, 'label': test_label}, ...],
                   3: [{'image': test_iamge, 'label': test_label}, ...],
                   ...},
               1: {0: [{'image': test_image, 'label': test_label}, ...],
                   2: [{'image': test_iamge, 'label': test_label}, ...],
                   ...}}
    :param test_images:
    :param test_labels:
    :param results:
    :return: results of successes and failures
    """
def _parse_test_results(test_images, test_labels, results):
    """Parse test results.

    Parse test results and split them into success and failure datasets.
    Both datasets are dict of list consisted of dict of image and label.
    successes: {0: [{'image': test_image, 'label': test_label}, ...],
                1: ...}
    failures: {0: {1: [{'image': test_image, 'label': test_label}, ...],
                   3: [{'image': test_iamge, 'label': test_label}, ...],
                   ...},
               1: {0: [{'image': test_image, 'label': test_label}, ...],
                   2: [{'image': test_iamge, 'label': test_label}, ...],
                   ...}}
    :param test_images:
    :param test_labels:
    :param results:
    :return: results of successes and failures
    """
    successes = {}
    failures = {}
    i = 0
    for batched_img, batched_lbl in zip(test_images, test_labels):
        # print(f'batch: {batched_img.shape} and label {batched_lbl.shape}')
        for j,img in enumerate(batched_img):
            test_image = img
            test_label = batched_lbl[j]
            test_label_index = test_label.numpy().argmax()

            result = results[i]
            # print(f'Print debug: img {test_image.shape}- lbl {test_label.shape} - label index {test_label_index} - result {result.shape}')
            predicted_label = result.argmax()
            if predicted_label != test_label_index:
                if test_label_index not in failures:
                    failures[test_label_index] = {}
                if predicted_label not in failures[test_label_index]:
                    failures[test_label_index][predicted_label] = []
                failures[test_label_index][predicted_label] \
                    .append({'image': test_image, 'label': test_label})
            else:
                if test_label_index not in successes:
                    successes[test_label_index] = []
                successes[test_label_index].append({'image': test_image, 'label': test_label})
            i += 1
    print(f'total: {i}')
    l = []
    for i in successes.keys():
        l.append(len(successes[i]))
        if i == 0:
            print(f'Found 0 class')
    print(f'Success: {np.array(l).sum()}')
    return successes, failures



def _save_label_data(data, path):
    """Save label data.

    Create `labels.json` for Athena and save it to given path.
    :param data: Dict consists of labes: list of data
    :param path: Path to save `labels.json`
    :return:
    """
    dict = {}
    for label in data:
        dict[str(label)] = {
            'repair_priority': 0,
            'prevent_degradation': 0,
        }
    with open(path, 'w') as f:
        dict_sorted = sorted(dict.items(), key=lambda x: x[0])
        json.dump(dict_sorted, f, indent=4)


def _create_merged_dataset(dataset):
    """Crerate merged dataset.

    Create merged dataset with all labels.
    given: {0: [{'image': image, 'label': label}, ...],
            1: [{'image': image, 'label': label}, ...],
            ...}
    return [image, ...], [label, ...]
    :param dataset: List of dataset grouped by labels
    :type dataset: list[]
    :return imgs, labels:
    """
    imgs = []
    labels = []
    for label in dataset:
        dataset_per_label = dataset[label]
        for data in dataset_per_label:
            imgs.append(data['image'])
            labels.append(data['label'])
    return imgs, labels


def _save_dataset_as_hdf5(images, labels, path):
    """Save datasets.

    create hdf5 file with given images and labels,
    then save it to given path.
    :param path:
    :param images:
    :param labels:
    """
    with h5py.File(path, 'w') as hf:
        hf.create_dataset('images',
                          data=np.array(images, dtype='float32'))
        hf.create_dataset('labels',
                          data=np.array(labels, dtype='float32'))


def _extract_dataset(dataset):
    """Extract images and labels from datatset.

    :param dataset: List of data. Data consists of image and its label.
    :type dataset: list[dict[list, list]]
    :returns: List of extracted images and labels
    :rtype: tupple(list, list)
    """
    images = []
    labels = []
    for result in dataset:
        image = result['image']
        label = result['label']
        images.append(image)
        labels.append(label)
    return images, labels


def _save_test_result(results, path):
    """Save result for single label dataset to given path.

    :param results: List of data.
    :type results: list[dict[list, list]]
    :param path: Path to save `repair.h5`
    :return: None
    """
    images, labels = _extract_dataset(results)
    _save_dataset_as_hdf5(images, labels, path)


def _save_test_results(results, data_dir, dataset_type='repair'):
    """Save results for multi labels' dataset to given path.

    given: {0: [{'image': image, 'label': label}, ...],
            1: [{'image': image, 'label': label}, ...],
            ...}
    Save to `data_dir`/<label>/repair.h5.
    :param results: List of data grouped by test label
    :type results: dict[int, list[dict[list, list]]]
    :param data_dir: Path to directory to save
    :return: None
    """
    for test_label in results:
        # Make directory for each class
        output_dir = data_dir \
            .joinpath((r'{}/'+f'{dataset_type}.h5').format(test_label))
        if not output_dir.parent.exists():
            output_dir.parent.mkdir(parents=True)

        _save_test_result(results[test_label], output_dir)


def _cleanup_dir(path, do_cleanup):
    """Clean up under given directory.

    :param path: Path to clean up directory
    """
    print(f"doing cleanup of path {path}")
    if path.exists():
        if do_cleanup:
            shutil.rmtree(path)
            path.mkdir()
    else:
        path.mkdir()


def _save_positive_results(results, data_dir, path, dataset_type='repair', do_cleanup=True):
    """Save positive data.

    :param results:
    :param data_dir:
    :param path:
    """
    output_dir = data_dir.joinpath(path)
    _cleanup_dir(output_dir, do_cleanup)

    _save_test_results(results, output_dir, dataset_type)

    # create all-in-one dataset
    all_images, all_labels = _create_merged_dataset(results)
    _save_dataset_as_hdf5(all_images,
                          all_labels,
                          output_dir.joinpath(f'{dataset_type}.h5'))


def _save_negative_results(results, data_dir, path, dataset_type='repair', do_cleanup=True):
    """Save negative data.

    :param results:
    :param data_dir:
    :param path:
    """
    output_dir = data_dir.joinpath(path)
    _cleanup_dir(output_dir, do_cleanup)

    # create each labels repair.h5
    for test_label in results:
        test_label_dir = output_dir.joinpath(str(test_label))
        if not test_label_dir.exists():
            Path(test_label_dir).mkdir(parents=True, exist_ok=True)
        _save_test_results(results[test_label], test_label_dir, dataset_type)

        # create all-in-one dataset per test label
        images_per_test_label, labels_per_test_label = \
            _create_merged_dataset(results[test_label])
        _save_dataset_as_hdf5(images_per_test_label,
                              labels_per_test_label,
                              test_label_dir.joinpath(f'{dataset_type}.h5'))
        _save_label_data(results[test_label],
                         test_label_dir.joinpath('labels.json'))

    # create all-in-one dataset
    all_imgs = []
    all_labels = []
    for labels in results:
        _imgs, _labels = _create_merged_dataset(results[labels])
        all_imgs.extend(_imgs)
        all_labels.extend(_labels)
    _save_dataset_as_hdf5(all_imgs,
                          all_labels,
                          output_dir.joinpath(fr'{dataset_type}.h5'))


def load_model(model_dir):
    """Load model.

    First, try to lead the model with the tensorflow standard function. If the model was saved with our different
    implementation, load it the previous way to save it with the standard tensorflow function.

    :param model_dir:
    :return:
    """
    try:
        model = tf.keras.models.load_model(model_dir)
        return model
    except Exception as e:
        print(f"exception while tensorflow was loading the model, trying now to manually load the model: " + str(e))
        model_dir = Path(model_dir)
        with open(model_dir.joinpath(r'model.json'), 'r') as json_file:
            model_json = json_file.read()
            custom_objects = _load_custom_objects('settings')
            with keras.utils.custom_object_scope(custom_objects):
                model = model_from_json(model_json)
                model.load_weights(model_dir.joinpath(r'model.h5'))
        model.compile(
            loss='categorical_crossentropy',
            optimizer='sgd',
            metrics=['accuracy']
        )
        # tf.keras.models.save_model(model, model_dir)
        print("done loading the model")
    return model


def _load_dataset(data_dir, target):
    """Load test dataset."""
    dataset = eAIDataset.EAIDataset(str(data_dir.joinpath(target)))

    return dataset


def _load_custom_objects(settings_dir):
    """Load settings.

    Loading an instance of repair methods, datasets, or models
    Settings are in "settings.json" under a given directory.

    :param settings_dir: path to directory containing settings.json
    :return: dict for using load model.
    """
    importlib.invalidate_caches()
    custom_dict = {}
    settings_dir = Path(settings_dir)
    file = settings_dir.joinpath(r'settings.json')
    with open(file, 'r') as f:
        settings = json.load(f)
        if 'custom_objects' in settings:
            for c_name in settings['custom_objects']:
                c_class = settings['custom_objects'][c_name]
                _names = c_class.split('.')
                module_name = c_class.split('.' + _names[-1])[0]
                package_module = importlib.import_module(module_name)
                class_module = getattr(package_module, _names[-1])
                custom_dict[_names[-1]] = class_module
        return custom_dict
