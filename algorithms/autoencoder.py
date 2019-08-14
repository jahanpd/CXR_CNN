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
        self.a_paths = a_paths
        self.b_paths = b_paths
        self.c_paths = c_paths
        self.labels = labels

        if save_path is not None:
            self.save_path = save_path

        self.img_rows = 256
        self.img_cols = 256
        self.channels = 1

        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.optimizer = 'Adam'
        self.loss = 'binary_crossentropy'
        self.metrics = [tf.keras.metrics.AUC(),
                        tf.keras.metrics.FalseNegatives(),
                        tf.keras.metrics.FalsePositives(),
                        tf.keras.metrics.TrueNegatives(),
                        tf.keras.metrics.TruePositives()]

    def _preprocess_image(self, image):
        image = tf.image.decode_jpeg(image, channels = self.channels)
        image = tf.image.resize(image, [self.img_rows, self.img_cols])
        image /= 255.0  # normalize to [0,1] range
        return image

    def _load_and_preprocess_image(self, paths): # load from path and return tensor
        image = tf.io.read_file(paths)
        return self._preprocess_image(image)

    def _input_fn(self):
        a = tf.data.Dataset.from_tensor_slices((self.a_paths))
        a = a.map(self._load_and_preprocess_image, num_parallel_calls=self.AUTOTUNE)
        b = tf.data.Dataset.from_tensor_slices((self.b_paths))
        b = b.map(self._load_and_preprocess_image, num_parallel_calls=self.AUTOTUNE)
        c = tf.data.Dataset.from_tensor_slices((self.c_paths))
        c = c.map(self._load_and_preprocess_image, num_parallel_calls=self.AUTOTUNE)
        label =  tf.data.Dataset.from_tensor_slices((self.labels))
        dataset = tf.data.Dataset.zip(({"input_1": a, "input_2": b, "input_3": c}, label))
        dataset = dataset.shuffle(buffer_size=2000)
        dataset = dataset.batch(32).repeat()
        dataset = dataset.prefetch(buffer_size=self.AUTOTUNE)
        return dataset


    def _encoder(self, a, b, c, filters):
        F1, F2, F3, F4, F5 = filters

        l1 = Conv2D(filters = F1,
            kernel_size = (1, 1),
            strides = (1,1), padding = 'valid',
            kernel_initializer = glorot_uniform(seed=0))
        b1= BatchNormalization(axis = 3)
        a1 = Activation('relu')

        m1 = MaxPooling2D((2,2), strides=(2,2))

        l2 = Conv2D(filters = F2,
            kernel_size = (1, 1),
            strides = (1,1), padding = 'valid',
            kernel_initializer = glorot_uniform(seed=0))
        b2= BatchNormalization(axis = 3)
        a2 = Activation('relu')

        m2 = MaxPooling2D((2,2), strides=(2,2))

        l3 = Conv2D(filters = F3,
            kernel_size = (1, 1),
            strides = (1,1), padding = 'valid',
            kernel_initializer = glorot_uniform(seed=0))
        b3= BatchNormalization(axis = 3)
        a3 = Activation('relu')

        m3 = MaxPooling2D((2,2), strides=(2,2))

        l4 = Conv2D(filters = F4,
            kernel_size = (1, 1),
            strides = (1,1), padding = 'valid',
            kernel_initializer = glorot_uniform(seed=0))
        b4= BatchNormalization(axis = 3)
        a4 = Activation('relu')

        m4 = MaxPooling2D((2,2), strides=(2,2))

        l5 = Conv2D(filters = F5,
            kernel_size = (1, 1),
            strides = (1,1), padding = 'valid',
            kernel_initializer = glorot_uniform(seed=0))
        b5 = BatchNormalization(axis = 3)
        a5 = Activation('relu')

        a,b,c = l1(a),l1(b),l1(c)
        a,b,c = b1(a),b1(b),b1(c)
        a,b,c = a1(a),a1(b),a1(c)
        a,b,c = m1(a),m1(b),m1(c)

        a,b,c = l2(a),l2(b),l2(c)
        a,b,c = b2(a),b2(b),b2(c)
        a,b,c = a2(a),a2(b),a2(c)
        a,b,c = m2(a),m2(b),m2(c)

        a,b,c = l3(a),l3(b),l3(c)
        a,b,c = b3(a),b3(b),b3(c)
        a,b,c = a3(a),a3(b),a3(c)
        a,b,c = m3(a),m3(b),m3(c)

        a,b,c = l4(a),l4(b),l4(c)
        a,b,c = b4(a),b4(b),b4(c)
        a,b,c = a4(a),a4(b),a4(c)
        a,b,c = m4(a),m4(b),m4(c)

        a,b,c = l5(a),l5(b),l5(c)
        a,b,c = b5(a),b5(b),b5(c)
        a,b,c = a5(a),a5(b),a5(c)

        return a,b,c

    def _decoder(self, a, b, c, filters):
        F1, F2, F3, F4, F5 = filters

        l1 = Conv2DTranspose(filters = F1,
            kernel_size = (1, 1),
            strides = (1,1), padding = 'valid',
            kernel_initializer = glorot_uniform(seed=0))
        b1 = BatchNormalization(axis = 3)
        a1 = Activation('relu')

        u1 = UpSampling2D()

        l2 = Conv2DTranspose(filters = F2,
            kernel_size = (1, 1),
            strides = (1,1), padding = 'valid',
            kernel_initializer = glorot_uniform(seed=0))
        b2 = BatchNormalization(axis = 3)
        a2 = Activation('relu')

        u2 = UpSampling2D()

        l3 = Conv2DTranspose(filters = F3,
            kernel_size = (1, 1),
            strides = (1,1), padding = 'valid',
            kernel_initializer = glorot_uniform(seed=0))
        b3 = BatchNormalization(axis = 3)
        a3 = Activation('relu')

        u3 = UpSampling2D()

        l4 = Conv2DTranspose(filters = F4,
            kernel_size = (1, 1),
            strides = (1,1), padding = 'valid',
            kernel_initializer = glorot_uniform(seed=0))
        b4 = BatchNormalization(axis = 3)
        a4 = Activation('relu')

        u4 = UpSampling2D()

        l5 = Conv2DTranspose(filters = F5,
            kernel_size = (1, 1),
            strides = (1,1), padding = 'valid',
            kernel_initializer = glorot_uniform(seed=0))
        b5 = BatchNormalization(axis = 3)
        a5 = Activation('relu')

        flat = Flatten()

        a,b,c = l1(a),l1(b),l1(c)
        a,b,c = b1(a),b1(b),b1(c)
        a,b,c = a1(a),a1(b),a1(c)
        a,b,c = u1(a),u1(b),u1(c)

        a,b,c = l2(a),l2(b),l2(c)
        a,b,c = b2(a),b2(b),b2(c)
        a,b,c = a2(a),a2(b),a2(c)
        a,b,c = u2(a),u2(b),u2(c)

        a,b,c = l3(a),l3(b),l3(c)
        a,b,c = b3(a),b3(b),b3(c)
        a,b,c = a3(a),a3(b),a3(c)
        a,b,c = u3(a),u3(b),u3(c)

        a,b,c = l4(a),l4(b),l4(c)
        a,b,c = b4(a),b4(b),b4(c)
        a,b,c = a4(a),a4(b),a4(c)
        a,b,c = u4(a),u4(b),u4(c)

        a,b,c = l5(a),l5(b),l5(c)
        a,b,c = b5(a),b5(b),b5(c)
        a,b,c = a5(a),a5(b),a5(c)

        a,b,c = flat(a),flat(b),flat(c)

        return a,b,c


    def build(self, shape=None, load=None):
        self.shape = (self.img_rows,self.img_cols,self.channels)
        if shape is not None:
            self.shape = shape

        # build network
        input_a = Input(shape=self.shape)
        input_b = Input(shape=self.shape)
        input_c = Input(shape=self.shape)

        # encode

        a,b,c = self._encoder(input_a,input_b,input_c, [16,32,64,64,32])

        # decode
        a,b,c = self._decoder(a,b,c, [32,64,32,16,1])



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
        if load is not None:
            self.nn.load_weights(self.save_path)
        history = self.nn.fit(self._input_fn(), epochs=epoch, steps_per_epoch=5000)
        self.nn.save(self.save_path+"ae.h5")
        return history

    def predict(self,test_paths,labels):
        path_ds = tf.data.Dataset.from_tensor_slices(test_paths)
        image_ds = path_ds.map(self._load_and_preprocess_image,
                                num_parallel_calls=self.AUTOTUNE)
        label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int16))
        ds = tf.data.Dataset.zip((image_ds, label_ds))
        ds = ds.batch(32)
        ds = ds.prefetch(buffer_size=self.AUTOTUNE)
        predictions = self.nn.predict(ds)
        return [self.nn.evaluate(ds),predictions]
