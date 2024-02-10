"""Modules for training the DNN model."""

from pathlib import Path

import importlib
import h5py
import json
import numpy as np
import tensorflow as tf
import scipy as sp

from tensorflow import keras
from keras.backend import set_session
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard, EarlyStopping
from keras.models import model_from_json

from tensorflow.keras.optimizers import SGD
from dataset import eAIDataset


def train(model,
          input_shape,
          classes,
          batch_size=32,
          epochs=50,
          validation_split=0.2,
          gpu=False,
          data_dir=r'outputs/',
          output_dir=r'outputs/',
          weighted_classes=False,
          ):
    """Train.

    :param model:
    :param input_shape:
    :param classes:
    :param batch_size:
    :param epochs:
    :param validation_split:
    :param gpu: configure GPU settings
    :param data_dir:
    :param output_dir:
    :param weighted_classes: If true, give more importance to some classes during training, depending on settings/classes_weights.py.json
    :return:
    """
    # GPU settings
    if gpu:
        config = tf.compat.v1.ConfigProto(
            gpu_options=tf.compat.v1.GPUOptions(
                allow_growth=True,
                per_process_gpu_memory_fraction=0.9)
        )
        session = tf.compat.v1.Session(config=config)
        set_session(session)

    current_dir = Path(output_dir)

    # callbacks
    mc_path = current_dir.joinpath(r'logs/model_check_points')
    mc_path.mkdir(parents=True, exist_ok=True)
    tb_path = current_dir.joinpath(r'logs/tensor_boards')
    tb_path.mkdir(parents=True, exist_ok=True)

    weight_path = mc_path.joinpath('weights.{epoch:02d}-{val_loss:.2f}.hdf5')
    model_check_point = ModelCheckpoint(filepath=str(weight_path),
                    verbose=1, 
                    save_freq='epoch',
                    save_weights_only=False,
                    save_best_only=False)

    tensor_board = TensorBoard(log_dir=str(tb_path))
    early_stop = EarlyStopping(monitor="val_loss", patience=5)

    lr_sc = LearningRateScheduler(__lr_schedule)

    callbacks = [model_check_point, tensor_board, lr_sc]
    dataset = eAIDataset.EAIDataset(data_dir + "train.h5")

    # Load Model
    try:
        model = model.compile(dataset.image_shape[1:], dataset.label_shape[1])
    except IndexError:
        # The case of training non one-hot vector
        model = model.compile(dataset.image_shape[1:], 1)
    # model.summary() # NOTE modified here

    # generates the inputs and labels in batches, with shuffling and validation set
    images_train, labels_trian, images_validation, labels_validation = dataset.get_generators_split_validation()
    train_generator = tf.data.Dataset.zip((images_train, labels_trian))
    validation_generator = tf.data.Dataset.zip((images_validation, labels_validation))

    
    model.fit(train_generator,
              epochs=epochs,
              callbacks=callbacks,
              validation_data=validation_generator,
              class_weight=None,
              verbose=1,                        # NOTE modified here
              
              )

    model_json = model.to_json()
    with open(Path(output_dir).joinpath(r'model.json'), 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(Path(output_dir).joinpath(r'model.h5'))

    tf.keras.models.save_model(model,Path(output_dir).joinpath(r'entire_model.h5'))
    
def load_model(model_dir):
    """Load model WITH WEIGHTED LOSS.

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
            optimizer=SGD(lr=1e-4, momentum=0.9),
            metrics=['accuracy']
        )
        # tf.keras.models.save_model(model, model_dir)
        print("done loading the model")
    return model

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


def __lr_schedule(epoch, lr=0.01):
    return lr * (0.1 ** int(epoch / 10))
