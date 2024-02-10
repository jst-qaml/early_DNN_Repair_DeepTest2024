"""The German Traffic Sign Recognition Benchmark (GTSRB).

cf. https://github.com/wakamezake/gtrsb
"""

import json
import os
from pathlib import Path

from .. import dataset, test
from . import prepare


class GTSRB(dataset.Dataset):
    """API for DNN with GTSRB."""

    def prepare(self, input_dir, output_dir, divide_rate, random_state):
        """Prepare.

        :param input_dir:
        :param output_dir:
        :param divide_rate:
        :param random_state:
        :return:
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        print(output_dir)

        # Make output directory if not exist
        try:
            os.makedirs(output_dir)
        except FileExistsError:
            pass

        prepare.prepare(input_dir, output_dir, divide_rate, random_state)

        return

    def _get_input_shape(self):
        """Set the input_shape and classes of BDD."""
        return (32, 32, 3), 43

    def get_images_to_draw_figure(self, model_dir, data_dir, target_data=43):
        """Test and get success and failure images.

        :param model_dir:
        :param data_dir:
        :param target_data:
        :return:
        """
        model = self.load_model(model_dir)
        test_images, test_labels = self.load_test_data(data_dir)

        results = model.predict(test_images, verbose=0)

        success_data = []
        failure_data = []

        for i, image in enumerate(test_images):
            if results[i].argmax() == test_labels[i].argmax():
                success_data.append({
                    'image': image,
                    'label': test_labels[i].argmax()
                })
            else:
                failure_data.append({
                    'image': image,
                    'label': test_labels[i].argmax()
                })

        if target_data == 43:
            success_images = [[] for _ in range(target_data)]
            failure_images = [[] for _ in range(target_data)]
            for i, d in enumerate(success_data):
                success_images[d['label']].append(d['image'])
            for i, d in enumerate(failure_data):
                failure_images[d['label']].append(d['image'])
            return success_images, failure_images
        else:
            success_images = []
            failure_images = []
            for i, d in enumerate(success_data):
                if d['label'] == target_data:
                    success_images.append(d['image'])
            for i, d in enumerate(failure_data):
                if d['label'] == target_data:
                    failure_images.append(d['image'])
            return success_images, failure_images

    def get_test_results(self,
                         base_model_dir,
                         repaired_model_dir,
                         data_dir,
                         output_dir,
                         classes=43):
        """Get test results.

        :param base_model_dir:
        :param repaired_model_dir:
        :param data_dir:
        :param output_dir:
        :param classes:
        :return:
        """
        dict = {}

        # Evaluate base model accuracy
        base_model_dir = Path(base_model_dir)
        data_dir = Path(data_dir)
        base_model_accuracy = test.test(base_model_dir, data_dir)
        dict['base_model_accuracy'] = format(
            base_model_accuracy[1]*100,
            '.2f'
        )

        # Evaluate repaired model accuracy
        repaired_model_dir = Path(repaired_model_dir)
        repaired_model_accuracy = test.test(repaired_model_dir, data_dir)
        dict['repaired_model_accuracy'] = format(
            repaired_model_accuracy[1]*100,
            '.2f'
        )

        # Evaluate model by label
        base_model = self.load_model(base_model_dir)
        test_images, test_labels = self.load_test_data(data_dir)

        for test_label in test_labels:
            key = test_label.argmax()
            dict[str(key)] = {}

        results = base_model.predict(test_images, verbose=0)
        parse_results = self._parse_results(test_images, test_labels, results)

        for result in parse_results:
            dict[result['key']]['base_score'] = result['score']

        # Evaluate repaired model by label
        repaired_model = self.load_model(repaired_model_dir)

        results = repaired_model.predict(test_images, verbose=0)
        parse_results = self._parse_results(test_images, test_labels, results)

        for result in parse_results:
            dict[result['key']]['repaired_score'] = result['score']

        # Evaluate repaired model using success and failure data for each label
        for i in range(classes):
            pos_data_dir = data_dir.joinpath(r'positive/%d/' % (i))
            neg_data_dir = data_dir.joinpath(r'negative/%d/' % (i))
            try:
                pos_accuracy = test.test(repaired_model_dir, pos_data_dir)
                dict[str(i)]['success'] = format(pos_accuracy[1]*100, '.2f')
            except OSError:
                dict[str(i)]['success'] = 0.00
                pass
            try:
                neg_accuracy = test.test(repaired_model_dir, neg_data_dir)
                dict[str(i)]['failure'] = format(neg_accuracy[1]*100, '.2f')
            except OSError:
                dict[str(i)]['failure'] = 0.00
                pass

        # Save
        output_dir = Path(output_dir)
        with open(output_dir.joinpath(r'results.json'), 'w') as f:
            dict_sorted = sorted(dict.items(), key=lambda x: x[0])
            json.dump(dict_sorted, f, indent=4)

        return

    def _parse_results(self, test_images, test_labels, results):
        """Parse results."""
        count_dict = {}
        parse_results = []

        for test_label in test_labels:
            key = test_label.argmax()
            count_dict[str(key)] = {'success': 0, 'failure': 0}

        for i in range(len(test_labels)):
            test_label = test_labels[i:i + 1]
            test_label_index = test_label.argmax()

            result = results[i:i + 1]

            if result.argmax() == test_label_index:
                current = count_dict[str(test_label_index)]['success']
                count_dict[str(test_label_index)]['success'] = current + 1
            else:
                current = count_dict[str(test_label_index)]['failure']
                count_dict[str(test_label_index)]['failure'] = current + 1

        for key in count_dict:
            success = count_dict[key]['success']
            failure = count_dict[key]['failure']
            score = (success * 100) / (success + failure)
            parse_results.append({
                'key': key,
                'score': format(score, '.2f')
            })

        return parse_results
