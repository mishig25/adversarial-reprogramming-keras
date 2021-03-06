import numpy as np
import tensorflow as tf

from keras.models import Model
from keras.engine.topology import Layer
from keras import backend as K
from keras.layers import Input
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.optimizers import Adam

class AdvReprogramLayer(Layer):    
    """ A custom Keras layer for Adversarial Reprogramming.

    This model creates weights that have same shape as 
    an input image to imagenet models.
    As the model trains, these weights should be optimized 
    to 'trick' the model into desired intention.
    """
    
    def __init__(self, **kwargs):
        super(AdvReprogramLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """Builds and initializes weights of the layer.

        Args: 
            input_shape: A tuple indicating shape of the input to the layer.
        """
        img_shape = (1,299,299,3)
        self.adv_weights = self.add_weight(name = 'kernel', 
                                      shape = img_shape,
                                      initializer = 'uniform',
                                      trainable = True)
        super(AdvReprogramLayer, self).build(input_shape) 

    def call(self, x):
        """Combines MNIST image with the layer's weights
        to create an input for an imagenet model.

        Args: 
            x: A tensor input to the layer.
        """
        input_mask = np.pad(np.zeros([1, 28, 28, 3]),
                            [[0,0], [136, 135], [136, 135], [0,0]],
                            'constant', constant_values = 1)
        mask = tf.constant(input_mask, dtype = tf.float32)
        channel_image = tf.concat([x, x, x], axis = -1)
        rgb_image = tf.pad(tf.concat([x, x, x], axis = -1), 
        paddings = tf.constant([[0,0], [136, 135], [136, 135], [0,0]]))
        adv_image = tf.nn.tanh(tf.multiply(self.adv_weights, mask)) + rgb_image
        self.out_shape = adv_image.shape
        return adv_image

    def compute_output_shape(self, input_shape):
        """Returns output shape of the layer.

        Args: 
            input_shape: A tuple indicating shape of the input to the layer.
        """
        return self.out_shape

class AdvReprogramModel(object):
    """ A model that demonstrates Adversarial Reprogramming.

    This model used Keras Functional API and 
    `AdvReprogramLayer` (custom keras layer)  
    to adversarially reporgram imagenet model 
    into one that can classify MNIST digits.
    """

    def __init__(self):
        self.optimizer = Adam(lr=0.05, decay=0.96)
        self.model = self.build_model(summary = True)
    
    def build_model(self, summary = False):
        """ Builds Keras model for Adversarial Reprogramming.

        Args:
            summary: A boolean indicating whether to show model summary after its built.
        """

        mnist_shape = (28,28,1)
        inputs = Input(shape=mnist_shape)
        adv = AdvReprogramLayer()(inputs)
        inception = InceptionV3(weights='imagenet')
        inception.trainable = False
        outputs = inception(adv)
        model = Model(inputs=[inputs],outputs=[outputs])
        if summary: model.summary()
        model.compile(optimizer=self.optimizer,
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
        return model
    
    def train(self, epochs=100, batch_size=50):
        """Trains the model.

        Args:
            epochs: An integer number of epochs.
            batch_size: An integer indicating size of the batches.
        """
        (x_train, y_train), (x_test, y_test) = self.get_mnist()
        self.model.fit(x_train,y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_test, y_test))
    
    def get_mnist(self):
        """Gets, processes, and return MNIST dataset.

        Returns:
            A tuple of tuples that contain training and test data.
        """
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = (x_train - 0.5)*2.0
        x_test = (x_test - 0.5)*2.0
        x_train = np.expand_dims(x_train, axis=3)
        x_test = np.expand_dims(x_test, axis=3)
        y_train = to_categorical(y_train, num_classes=1000)
        y_test = to_categorical(y_test, num_classes=1000)
        return (x_train, y_train), (x_test, y_test)