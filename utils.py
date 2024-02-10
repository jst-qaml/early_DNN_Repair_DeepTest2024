import importlib
import json
import os
import sys
import collections
import subprocess as sp
import numpy as np

from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, confusion_matrix


def load_model(model_dir):
    """Load model.

    First, try to lead the model with the tensorflow standard function. If the model was saved with our different
    implementation, load it the previous way to save it with the standard tensorflow function.

    :param model_dir:
    :return:
    """
    import tensorflow as tf
    from tensorflow import keras
    from keras.models import model_from_json

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
        tf.keras.models.save_model(model, model_dir)
        print("done saving the model")
    return model

def _load_dataset(data_dir, target):
    """Load test dataset."""
    from dataset import eAIDataset
    dataset = eAIDataset.EAIDataset(str(data_dir.joinpath(target)))

    return dataset

def _get_dataset(name, **kwargs):
    """Get dataset.
    :param name: identifier of dataset
    :param kwargs: extra config available for specific dataset
           (e.g. target_label for BDD dataset)
    :return: dataset
    """
    print(f'dataset name: {name}')
    dataset = _load_instance('dataset', 'dataset', name, 'settings')
    if dataset is None:
        raise Exception('Invalid dataset: {}'.format(name))
    dataset.set_extra_config(kwargs)
    return dataset


def _get_model(name, **kwargs):
    """Get model.
    :param name: identifier of model
    :return: model
    """
    import tensorflow as tf
    model = _load_instance('model', 'model', name, 'settings')
    if model is None:
        raise Exception('Invalid model: {}'.format(name))
    model.set_extra_config(kwargs)
    return model


def _get_repair_method(name, dataset, kwargs):
    """Get repair method.
    :param name: identifier of repair method
    :param dataset: identifier of dataset
    :param kwargs: repair method options
    :return: repair method
    """
    # Instantiate repair method
    method = _load_instance('method', 'repair', name, 'settings')
    if method is None:
        raise Exception('Invalid method: {}'.format(name))
    # Set optional parameters
    method.set_options(dataset, kwargs)

    return method


def _load_instance(key, parent_module, name, settings_dir):
    """Load settings.
    Loading a instance of repair methods, datasets, or models
    Settings are in "settings.json" under a given directory.
    :param key: key of settings.json (e.g. method)
    :param parent_module: directory containing modules (e.g. repair)
    :param name: available option name (e.g. Arachne for method)
    :param settings_dir: path to directory containing settings.json
    :return: instance of repair methods, datasets, or models
    """
    importlib.invalidate_caches()
    settings_dir = Path(settings_dir)
    file = settings_dir.joinpath(r'settings.json')

    with open(file, 'r') as f:
        try:
            settings = json.load(f)
            if name in settings[key]:
                name = settings[key][name]
            _names = name.split('.')
            module_name = name.split('.' + _names[-1])[0]
            methodclass = importlib.import_module('.' + module_name,
                                                  parent_module)
            instance = getattr(methodclass, _names[-1])(_names[0])

            return instance
        except Exception as e:
            print(e)
            print('...')
            return None
        
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

        
def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


##########################################################
####################### Metrics ##########################
##########################################################

def get_metrics(model, test_data_path, target_data_dir, positive_data_dir, repaired_model):

    from dataset import eAIDataset

    test_dataset = eAIDataset.EAIDataset(test_data_path)
    images_generator, labels_generator = test_dataset.get_generators()
    print(f'image generator shape: {test_dataset.image_shape}')
    predictions = model.predict(images_generator, verbose=0)

    metrics_dict = {}

    pred_history = []
    true_history = []

    # Gather results
    index = 0
    for lbl_batch in labels_generator:
        for lbl in lbl_batch:
            true_history.append(lbl)

            prediction = predictions[index]
            pred_history.append(prediction)
            
            index += 1


    true_history = np.array(true_history)
    pred_history = np.array(pred_history)

    true_argmax = np.argmax(true_history, axis=1)
    pred_argmax = np.argmax(pred_history, axis=1)

    print(pred_history)

    # Calculate accuracy for each class
    per_class_accuracy = confusion_matrix(true_argmax, pred_argmax, normalize='true').diagonal()
    metrics_dict['per_class_accuracy'] = list(per_class_accuracy)
    print(f'per_class_accuracy: {per_class_accuracy}')

    # Calculate general metrics
    accuracy = accuracy_score(true_argmax, pred_argmax)
    f1 = f1_score(true_argmax, pred_argmax, average='macro')
    recall = recall_score(true_argmax, pred_argmax, average='macro')
    auc_under_roc = roc_auc_score(true_history, pred_history, average='macro')

    # Weighted accuracy calculation

    weights = np.where(true_argmax == 0, 1.5, 1)
    weighted_acc = accuracy_score(true_argmax, pred_argmax, sample_weight=weights)

    if repaired_model:
        # Also get RR and BR of model
        # Compute RR
        repair_images, repair_labels = eAIDataset.EAIDataset(target_data_dir+'repair.h5', batch_size=2).get_generators()
        # repair_images, repair_labels = eAIDataset.EAIDataset(test_data_path, batch_size=16).get_generators()
        predictions = model.predict(repair_images, verbose=0)
        print(len(predictions))
        pred_history = []
        true_history = []

        # Gather results
        index = 0
        for lbl_batch in repair_labels:
            for lbl in lbl_batch:
                true_history.append(lbl)

                prediction = predictions[index]
                pred_history.append(prediction)
                
                index += 1
        

        true_argmax = np.argmax(true_history, axis=1)
        pred_argmax = np.argmax(pred_history, axis=1)

        patched = (
            true_argmax == pred_argmax
        )
        patched = patched.sum()
        print(f'patched: {patched} and len true: {len(true_history)}')
        rr = patched / len(true_history) * 100

        repair_images, repair_labels = eAIDataset.EAIDataset(positive_data_dir+'repair.h5', batch_size=2).get_generators()
        # repair_images, repair_labels = eAIDataset.EAIDataset(test_data_path, batch_size=16).get_generators()
        predictions = model.predict(repair_images, verbose=0)
        pred_history = []
        true_history = []

        # Gather results
        index = 0
        for lbl_batch in repair_labels:
            for lbl in lbl_batch:
                true_history.append(lbl)

                prediction = predictions[index]
                pred_history.append(prediction)
                
                index += 1


        true_argmax = np.argmax(true_history, axis=1)
        pred_argmax = np.argmax(pred_history, axis=1)

        broken = (
            true_argmax
            != pred_argmax
        ).sum()
        print(f'broken: {broken} and len true: {len(true_history)}')
        br = broken / len(true_history) * 100

        metrics_dict['rr'] = rr
        metrics_dict['br'] = br


    metrics_dict['accuracy'] = accuracy
    metrics_dict['f1'] = f1
    metrics_dict['recall'] = recall
    metrics_dict['auc_under_roc'] = auc_under_roc
    metrics_dict['weighted_acc'] = weighted_acc
    

    return metrics_dict