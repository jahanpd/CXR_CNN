import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from sklearn.metrics import roc_auc_score as auc

# tf.enable_eager_execution()

class AUC(keras.metrics.Metric):
    def __init__(self, name='categorical_true_positives', **kwargs):
      super(AUC, self).__init__(name=name, **kwargs)
      self.auc = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
      values = tf.cast(auc(np.array(y_true), np.array(y_pred)), 'float32')
      if sample_weight is not None:
        sample_weight = tf.cast(sample_weight, 'float32')
        values = tf.multiply(values, sample_weight)
      self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
      return self.auc

    def reset_states(self):
      # The state of the metric will be reset at the start of each epoch.
      self.true_positives.assign(0.)


# custom euclidian distance output layer
# https://www.sciencedirect.com/science/article/pii/S2212827119302409
class Euclidian(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Euclidian, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.
        super(Euclidian, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        assert isinstance(x, list)
        a, b = x # a is input later and b is output layer
        return K.sqrt(K.sum(K.square(b - a), axis=-1))

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0], self.output_dim), shape_b[:-1]]
