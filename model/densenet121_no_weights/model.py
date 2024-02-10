"""ResNet50 fine tuning model."""
from tensorflow.keras.applications.densenet import DenseNet121
from keras.layers import Input
from keras.models import Model
from tensorflow.keras.optimizers import SGD

from .. import model


class DenseNet121Model(model.Model):
    """DenseNet121 from scratch"""

    def compile(self, input_shape, output_shape):
        """Configure ResNet50 model.

        :param input_shape:
        :param output_shape:
        :return: model
        """
        # Use VGG16 as basis
        densenet121_model = DenseNet121(
                include_top=True,
                weights=None,
                input_shape=input_shape,
                input_tensor=Input(shape=input_shape),
                classes=2,
                pooling=None,
            )

        model = Model(inputs=densenet121_model.input, outputs=densenet121_model.output)

        # print(model.summary())

        # Compile model
        model.compile(optimizer=SGD(lr=1e-4, momentum=0.9),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model


