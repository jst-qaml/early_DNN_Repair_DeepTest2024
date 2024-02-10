"""Prepare ImageNette."""

import numpy as np
import pandas as pd
import pydicom as dcm
import os
from skimage import transform
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
from ..prepare import save_prepared_data
from tensorflow.keras.utils import to_categorical

def prepare(root_path, output_path, divide_rate, random_state):
    """Prepare.

    :param output_path:
    :param divide_rate:
    :param random_state:
    :return:
    """
    train_images, train_labels, \
        repair_images, repair_labels, \
             test_images, test_labels = _get_train_test_images_and_labels(root_path, divide_rate, random_state)

    save_prepared_data(train_images,
                       train_labels,
                       repair_images,
                       repair_labels,
                       test_images,
                       test_labels,
                       output_path)


def _get_train_test_images_and_labels(
        root_path,
        divide_rate,
        random_state,
        target_size=(160, 160)):
    
    images = []
    labels = []
    test_images = []
    test_labels= []
    root_path = Path(root_path)

    # First get labels
    train_mapping_path = root_path.joinpath('stage_2_train_labels.csv')
    # test_mapping_path = root_path.joinpath('nih-cxr-lt_single-label_test.csv')

    train_img_to_class = _get_mapping(train_mapping_path)
    # test_img_to_class, test_second_max_num = _get_mapping(test_mapping_path)

    # Second get images
    all_image_paths = root_path.glob('stage_2_train_images/*.dcm')
    # print(f'image paths: {all_image_paths}')
    # lab = train_img_to_class['1792982c-8183-4199-8217-510c92805614']
    # print(f'lab: {lab}')

    for img_path in tqdm(all_image_paths, desc='Preproc'):
        img = _preprocess_img(dcm.dcmread(img_path).pixel_array, target_size)
        if img is None:
            print(f'Preproc image returned None')
            continue

        # Look for label in trian or test split
        file_name = os.path.basename(img_path)[:-4] 
        # print(f'fname: {file_name}')
        cur_label = _get_class(file_name, train_img_to_class)
        # print(f'Cur label: {cur_label}')

        images.append(img)
        labels.append(to_categorical(cur_label, num_classes=2))

    print(f'Done gathering images, total labeled-{len(images)}')

    # Get some test samples
    remain_images, test_images, remain_labels, test_labels = \
        train_test_split(images, labels, test_size=0.1, random_state=random_state+1, stratify=np.argmax(labels, axis=1))

    # Split again into train and repair
    train_images, repair_images, train_labels, repair_labels = \
        train_test_split(remain_images, remain_labels, test_size=0.05, random_state=random_state, stratify=np.argmax(remain_labels, axis=1))

    print(f'num repair imgs: {np.array(repair_images).shape}')

    return np.array(train_images, dtype='float32'), \
        np.array(train_labels, dtype='uint8'), \
        np.array(repair_images, dtype='float32'), \
        np.array(repair_labels, dtype='uint8'), \
        np.array(test_images, dtype='float32'), \
        np.array(test_labels, dtype='uint8'), 

def _get_class(img_name, mapping_dict):
    try:
        cls = mapping_dict[img_name]
    except KeyError as e:
        return None

    return cls

def _preprocess_img(img, target_size):
    
    # Image normalization?
    min, max = img.min(), img.max()
    if min < 0 or max > 255:
        print(f'Image is not in desired intensity range, rescale...')
        img = ((img - img.min()) * (1/(img.max() - img.min()) * 255)).astype('uint8')
        print(f'New image intensities: {img.min()} - {img.max()}')

    # rescale to standard size
    # if image is too small, just throw it away
    try:
        if len(img.shape) > 2:
            print(f"Image is not initially 2 dimensional, it has shape {img.shape}")
        img = transform.resize(img, target_size)
    except:
        return None
    
    img = img[:,:,np.newaxis]

    return img

def _get_mapping(csv_path):
    # Keep class of each image for fast processing
    mapping_img_to_number = {}

    df = pd.read_csv(csv_path)

    ids = df['patientId'].values
    # df.reset_index(drop=True)
    labels = df['Target'].values

    print(f'length of ids: ({len(ids)}) and labels ({len(labels)}) should be the same')

    argmax = labels
    unique, counts = np.unique(argmax, return_counts=True)
    # second_max_num = np.sort(counts.flatten())[-2]

    print(f'For file {csv_path} with  class balance:\n{np.asarray((unique, counts)).T}')

    for i in range(len(ids)):
        mapping_img_to_number[ids[i]] = labels[i]

    return mapping_img_to_number