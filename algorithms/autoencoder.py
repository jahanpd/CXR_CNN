import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Add, Dense, Activation, Softmax, BatchNormalization, Flatten, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
import tensorflow.keras.backend as K
from algorithms.custom import Euclidian
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from pathlib import Path

# tf.enable_eager_execution()

class autoencoder:
    def __init__(self, a_paths, b_paths, c_paths ,labels, save_path=None):
        K.set_floatx('float32')
        self.a_paths,self.b_paths,self.c_paths,self.labels = a_paths, b_paths,c_paths, labels

        if save_path is not None:
            self.save_path = save_path

        self.img_rows = 256
        self.img_cols = 256
        self.channels = 1

        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.optimizer = 'Adam'
        self.loss = 'binary_crossentropy'
        self.metrics = [tf.keras.metrics.BinaryCrossentropy(),
                        tf.keras.metrics.AUC(),
                        tf.keras.metrics.BinaryAccuracy()]
                        #tf.keras.metrics.FalseNegatives(),
                        #tf.keras.metrics.FalsePositives(),
                        #tf.keras.metrics.TrueNegatives(),
                        #tf.keras.metrics.TruePositives()]

    def _preprocess_image(self, image):
        image = tf.image.decode_jpeg(image, channels = self.channels)
        image = tf.image.resize(image, [self.img_rows, self.img_cols])
        image /= 255.0  # normalize to [0,1] range
        return image

    def _load_and_preprocess_image(self, paths): # load from path and return tensor
        image = tf.io.read_file(paths)
        return self._preprocess_image(image)

    def _input_fn(self, x_a,x_b,x_c, y):
        a = tf.data.Dataset.from_tensor_slices((x_a))
        a = a.map(self._load_and_preprocess_image, num_parallel_calls=self.AUTOTUNE)
        b = tf.data.Dataset.from_tensor_slices((x_b))
        b = b.map(self._load_and_preprocess_image, num_parallel_calls=self.AUTOTUNE)
        c = tf.data.Dataset.from_tensor_slices((x_c))
        c = c.map(self._load_and_preprocess_image, num_parallel_calls=self.AUTOTUNE)
        label =  tf.data.Dataset.from_tensor_slices((y))
        dataset = tf.data.Dataset.zip(({"input_1": a, "input_2": b, "input_3": c}, label))
        dataset = dataset.shuffle(buffer_size=100)
        dataset = dataset.batch(8).repeat()
        dataset = dataset.prefetch(buffer_size=self.AUTOTUNE)
        return dataset


    def _resblock(self, a, b, c, filters):
        F1, F2, F3, F4 = filters

        a_, b_, c_ = a, b, c

        l1 = Conv2D(filters = F1,
            kernel_size = (5, 5),
            strides = (1,1), padding = 'same',
            kernel_initializer = glorot_uniform(seed=0))
        b1= BatchNormalization(axis = 3)
        a1 = Activation('relu')

        l2 = Conv2D(filters = F2,
            kernel_size = (5, 5),
            strides = (1,1), padding = 'same',
            kernel_initializer = glorot_uniform(seed=0))
        b2= BatchNormalization(axis = 3)
        a2 = Activation('relu')

        l3 = Conv2D(filters = F3,
            kernel_size = (1, 1),
            strides = (1,1), padding = 'same',
            kernel_initializer = glorot_uniform(seed=0))
        b3= BatchNormalization(axis = 3)
        a3 = Activation('relu')

        l4 = Conv2D(filters = F4,
            kernel_size = (1, 1),
            strides = (1,1), padding = 'same',
            kernel_initializer = glorot_uniform(seed=0))
        b4= BatchNormalization(axis = 3)
        a4 = Activation('relu')

        add = Add()

        a,b,c = l1(a),l1(b),l1(c)
        a,b,c = b1(a),b1(b),b1(c)
        a,b,c = a1(a),a1(b),a1(c)

        a,b,c = l2(a),l2(b),l2(c)
        a,b,c = b2(a),b2(b),b2(c)
        a,b,c = a2(a),a2(b),a2(c)

        a,b,c = l3(a),l3(b),l3(c)
        a,b,c = b3(a),b3(b),b3(c)
        a,b,c = a3(a),a3(b),a3(c)

        a_, b_, c_ = l4(a_),l4(b_),l4(c_)
        a_, b_, c_  = b4(a_),b4(b_),b4(c_)

        a,b,c = add([a,a_]), add([b,b_]), add([c,c_])

        a,b,c = a4(a),a4(b),a4(c)

        return a,b,c


    def build(self, shape=None, load=None):
        self.shape = (self.img_rows,self.img_cols,self.channels)
        if shape is not None:
            self.shape = shape

        m = MaxPooling2D((2,2), strides=(2,2))
        flat = Flatten()

        # build network
        input_a = Input(shape=self.shape)
        input_b = Input(shape=self.shape)
        input_c = Input(shape=self.shape)

        # encode

        a,b,c = self._resblock(input_a,input_b,input_c, [16,16,16,16])
        a,b,c = m(a),m(b),m(c)
        a,b,c = self._resblock(a,b,c, [16,16,16,16])
        a,b,c = m(a),m(b),m(c)
        a,b,c = self._resblock(a,b,c, [16,16,16,16])
        a,b,c = m(a),m(b),m(c)
        a,b,c = self._resblock(a,b,c, [16,16,16,16])
        a,b,c = m(a),m(b),m(c)
        a,b,c = self._resblock(a,b,c, [32,32,32,32])

        a,b,c = flat(a), flat(b), flat(c)

        euclidian_ab = Euclidian(1)([a,b])
        euclidian_ac = Euclidian(1)([a,c])

        merged = tf.stack([euclidian_ab,euclidian_ac],axis=1)

        output = Softmax()(merged)

        self.nn = Model(inputs = [input_a,input_b,input_c], outputs = output)
        self.nn.compile(optimizer = self.optimizer,loss=self.loss, metrics=self.metrics)
        self.nn.summary()

        if load is not None:
            self.nn = load_model(self.save_path + "ae.h5")

    def train(self, epoch, load=None):
        cp_callback = tf.keras.callbacks.ModelCheckpoint(self.save_path+'triple.ckpt',
                                                 save_weights_only=True,
                                                 verbose=1)
        if load is not None:
            self.nn.load_weights(self.save_path + load)
        history = self.nn.fit(self._input_fn(self.a_paths,self.b_paths,self.c_paths,self.labels),
                            epochs=epoch, steps_per_epoch=8*100000,callbacks = [cp_callback])
        self.nn.save_weights(self.save_path + "model.weights")
        return history

    def predict(self,x_a,x_b,x_v, y):
        predictions = self.nn.predict(self._input_fn(x_a,x_b,x_v, y))
        return [self.nn.evaluate(self._input_fn(x_a,x_b,x_v, y)),predictions]
