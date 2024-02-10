"""Class EAIDataset

This class indicates is a wrapper for tfio.Dataset, adding some size and shape
hints we know when we open a hdf5 file. Also has methods for sampling some positive inputs as numpy array.
"""

import math

import numpy as np
import tensorflow as tf

import h5py


class EAIDataset:
    """class of EAIDataset."""

    def __init__(self, file_name, batch_size=4):
        """Initialize.

        :param file_name:
        :param batch_size:
        """
        hf = h5py.File(file_name, 'r')  # open the file only once
        self.hf = hf
        self.image_shape = hf["images"].shape  # shape of the images dataset
        self.label_shape = hf["labels"].shape
        self.image_type = hf["images"].dtype
        self.label_type = hf["labels"].dtype
        self.keys = list(hf.keys())  # names of the datasets in the file (should be images and labels)
        self.file_name = file_name
        self.batch_size = batch_size
        self.index_handler = None

        # indexes to be used for EAIIndexHandler.
        self.indices = np.arange(len(self))  # one index per batch
        np.random.shuffle(self.indices)

    def __len__(self):
        """number of times we can give next item, since each item is a batch,
        not a single image, we divide by batch_size"""

        assert math.ceil(self.amount_of_images() / self.batch_size) == (self.amount_of_images() / self.batch_size)
        return int(math.ceil(self.amount_of_images() / self.batch_size))

    def amount_of_images(self):
        if self.image_shape[0] % self.batch_size == 0:
            return self.image_shape[0]
        else:  # add the loop around remainder that we add so all batches have the same size
            return self.image_shape[0] + (self.batch_size - (self.image_shape[0] % self.batch_size))

    def close_file(self):  # close the h5 file when we don't need to read any more data
        self.hf.close()

    def get_image_number(self, idx):
        """interface to manually get the image with the chosen index.
        Warning, this is much slower than the generator, since it uses a batch size of 1 to access the file instead
        of the default 32."""
        return self.hf["images"][idx]

    def get_label_number(self, idx):
        """interface to manually get the label with the chosen index.
        Warning, this is much slower than the generator, since it uses a batch size of 1 to access the file instead
        of the default 32."""
        return self.hf["labels"][idx]

    def get_generators_split_validation(self, validation_size=0.2):
        """returns four generators. images_train, labels_train, images_validation, labels_validation.
        The images and labels datasets can be zipped together to use with model.fit"""
        validation_indices = self.indices[:int(math.floor(validation_size * len(self.indices)))]
        train_indices = self.indices[int(math.floor(validation_size * len(self.indices))):]
        train_index_handler = EAIIndexHandler(self.amount_of_images(), self.batch_size, train_indices)
        validation_index_handler = EAIIndexHandler(self.amount_of_images(), self.batch_size, validation_indices)

        images_train = self.get_images_generator(train_index_handler)
        labels_train = self.get_labels_generator(train_index_handler)
        images_validation = self.get_images_generator(validation_index_handler)
        labels_validation = self.get_labels_generator(validation_index_handler)

        return images_train, labels_train, images_validation, labels_validation

    def get_real_index(self, idx):
        """For when we want to know what is the real h5 file index for the input we return in position idx for the
        current shuffling"""
        batch = math.floor(idx / self.batch_size)
        batch = self.index_handler.indices[batch]
        ans = (batch * self.batch_size) + (idx % self.batch_size)
        return ans % self.image_shape[0]  # to avoid out of range because of the incomplete last batch


    def get_generators(self):
        """returns a pair of generators, one for images and one for labels. Note that this does not split the dataset
        for validation purposes during training.
        The images and labels datasets can be zipped together to use with model.fit"""

        self.index_handler = EAIIndexHandler(self.amount_of_images(), self.batch_size, self.indices)

        images_generator = self.get_images_generator(self.index_handler)
        labels_generator = self.get_labels_generator(self.index_handler)

        return images_generator, labels_generator

    def get_images_generator(self, index_handler):

        try:
            output_shapes=tf.TensorShape([self.batch_size, self.image_shape[1], self.image_shape[2],
                self.image_shape[3]])  # shape of the tensor that we generate: [batch_size,image_shape]
        # Handle the case for text data
        except:
            output_shapes=tf.TensorShape([self.batch_size, self.image_shape[1]])  # shape of the tensor that we generate: [batch_size,image_shape]

        images_ds = tf.data.Dataset.from_generator(
            EAIGenerator(index_handler, self.hf["images"]),
            output_types=self.image_type,
            output_shapes=output_shapes)  # shape of the tensor that we generate: [batch_size,image_shape]
        return images_ds

    def get_labels_generator(self, index_handler):
        images_ds = tf.data.Dataset.from_generator(
            EAIGenerator(index_handler, self.hf["labels"]),
            output_types=self.label_type,
            output_shapes=tf.TensorShape([self.batch_size, self.label_shape[1]]))
        return images_ds

    def get_tuple_generator(self):
        self.index_handler = EAIIndexHandler(self.amount_of_images(), self.batch_size, self.indices)
        generator = self.get_images_labels_tuple(self.index_handler)
        return generator

    def get_images_labels_tuple(self, index_handler):
        images_generator = EAIGenerator(index_handler, self.hf["images"], self.hf["labels"])
        return images_generator

    def sample_from_file(self, sample_amount, indices=None):
        """To be used when only some inputs from the dataset will be sampled for use.
        The random choice is done in a per image basis, not like the generator which randomizes per batch.
        The arrays are sorted per index, so the labels will not be shuffled.

        :param sample_amount: how many images to sample. Size of returned arrays.
        :param indices: if None, sample from the whole dataset. If it's a list, sample only from those indices
        :return: 2 numpy arrays, one for images and one for labels.
        """

        labels = {}  # divide the index of the inputs depending on their label
        batch_size = self.batch_size
        if indices is None:
            for batch_num in range(len(self)):
                if batch_num == len(self) - 1:  # to avoid out of range
                    current_labels = self.hf["labels"][batch_num * batch_size:]
                else:
                    current_labels = self.hf["labels"][batch_num * batch_size: (batch_num + 1) * batch_size]
                for idx, label in enumerate(np.argmax(current_labels, axis=1)):
                    correct_index = (batch_num*batch_size) + idx
                    if label in labels:
                        labels[label].append(correct_index)
                    else:
                        labels[label] = [correct_index]
            total_size = self.label_shape[0]
        else:  # if a list of indices was given, sample only from that list.
            for index in indices:
                label = np.argmax(self.get_label_number(index))
                if label == 12:  # TODO quickfix, remove if no longer doing exp2a
                    continue
                if label in labels:
                    labels[label].append(index)
                else:
                    labels[label] = [index]
            total_size = len(indices)

        # at least put as many images of each label, as if you put a third of an equitative distribution
        amount_per_label = {}
        total = 0
        defend_underrepresented_labels = 2/3  # hyperparameter, if bigger, more images of underrepresented labels
        labels_not_saturated = set(labels.keys())  # labels that can sample more images, they don't have few examples
        saturation_limit = 2  # how many times we allow images to be sampled
        for label in labels:
            amount_per_label[label] = int((sample_amount / len(labels)) * defend_underrepresented_labels)
            amount_per_label[label] += int(
                (sample_amount * (len(labels[label]) / total_size)) * (1 - defend_underrepresented_labels))

            # if we would repeat to many images, it's better to sample less from this label
            if amount_per_label[label] > (len(labels[label]) * saturation_limit):
                amount_per_label[label] = (len(labels[label]) * saturation_limit)
                labels_not_saturated.remove(label)
            total += amount_per_label[label]

        while total < sample_amount:
            current_set = labels_not_saturated.copy()
            if total - sample_amount < 10:
                amount_per_label[np.random.choice(list(current_set), 1)[0]] += sample_amount - total
                break
            for label in current_set:
                amount_per_label[label] += math.ceil((sample_amount - total) / len(labels_not_saturated))
                # if we would repeat to many images, it's better to sample less from this label
                if amount_per_label[label] > (len(labels[label]) * saturation_limit):
                    amount_per_label[label] = (len(labels[label]) * saturation_limit)
                    labels_not_saturated.remove(label)
                total += math.ceil((sample_amount - total) / len(labels_not_saturated))
                if len(labels_not_saturated) == 0:
                    assert False, "asked for a really big sample but there are few images to sample from," \
                                  " please sample less images or change the saturation_limit to allow more" \
                                  " images to repeat during the sampling"
        if total > sample_amount:
            print(f"warning, sampling {total - sample_amount} more images than requested")

        # actual sampling
        sample = np.empty(0, int)
        for label in amount_per_label:
            while amount_per_label[label] >= len(labels[label]):
                sample = np.concatenate((sample, labels[label]))
                amount_per_label[label] -= len(labels[label])
            sample = np.concatenate((sample, np.random.choice(labels[label], amount_per_label[label], replace=False)))

        assert len(sample) == sample_amount
        sample = np.unique(sample)  # apparently, duplicate indexes are bad for h5py fancy indexes.
        sample.sort()
        return self.hf["images"][sample], self.hf["labels"][sample]


class EAIIndexHandler:
    def __init__(self, image_amount, batch_size, indices):
        self.batch_size = batch_size
        self.indices = indices  # the indices we shuffle and use to access h5 file
        self.image_amount = image_amount  # number of images and labels in total in the h5 file.

    def shuffle(self):
        np.random.shuffle(self.indices)

    def __len__(self):
        """number of times we can give next item, since each item is a batch,
        not a single image, we divide by batch_size"""

        return int(math.ceil(self.image_amount / self.batch_size))


class EAIGenerator(tf.keras.utils.Sequence):
    """A generator to iterate over a h5 dataset. Never loads the whole dataset, so it can deal with OutOfMemory
    issues. Gets the images or labels in batches from the file, to make I/O faster, because of this, when we shuffle or access
    randomly, we access random batches, but that batch of images is always in the same order.
    Gets the indexes from EAI_Index_Handler, so images and labels are accessed in the same order."""

    def __init__(self, index_handler, dataset, targets=None):
        self.index_handler = index_handler
        self.dataset = dataset  # either "images" or "labels" dataset from the h5 file
        self.targets = targets

    def __getitem__(self, idx):
        """get batch Number idx from file. Note that accessing the h5 file with slices as done here
         is faster than providing a list or range of indices to fetch
        https://docs.h5py.org/en/latest/high/dataset.html#fancy-indexing"""
        batch_size = self.index_handler.batch_size
        if idx == len(self)-1:  # to avoid out of range
            # print(f'inside last batch')
            batch = self.dataset[idx * batch_size:]
            batch = np.concatenate((batch, self.dataset[:batch_size - len(batch)]))
        else:
            batch = self.dataset[idx * batch_size: (idx + 1) * batch_size]

        if self.targets is None:
            return batch
        
        # sample from targets also
        else:
            if idx == self.__len__()-1:  # to avoid out of range
                t_batch = self.targets[idx * batch_size:]
                t_batch = np.concatenate((t_batch, self.targets[:batch_size - len(t_batch)]))
            else:
                t_batch = self.targets[idx * batch_size: (idx + 1) * batch_size]
        return (batch, t_batch)

    def __call__(self):
        for idx in self.index_handler.indices:
            batch = self.__getitem__(idx)
            yield batch

    def on_epoch_end(self):
        print("on_epoch_end called. Shuffling the EAIGenerator.")
        self.index_handler.shuffle()

    def __len__(self):
        """number of times we can give next item, since each item is a batch,
        not a single image, we divide by batch_size"""

        return len(self.index_handler)
