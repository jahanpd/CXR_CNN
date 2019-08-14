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
    def __init__(self, image_paths, labels, save_path=None):
        K.set_floatx('float32')
        self.image_paths = image_paths
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

    def _load_and_preprocess_image(self, path): # load from path and return tensor
        image = tf.io.read_file(path)
        return self._preprocess_image(image)



    def build_dataset(self, batch_size=None):
        path_ds = tf.data.Dataset.from_tensor_slices(self.image_paths)
        image_ds = path_ds.map(self._load_and_preprocess_image,
                                num_parallel_calls=self.AUTOTUNE)
        label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(self.labels, tf.int16))
        image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
        self.ds = image_label_ds.shuffle(buffer_size=2000)
        if batch_size is not None:
            self.ds = self.ds.batch(batch_size)
        else:
            self.ds = self.ds.batch(32)
        self.ds = self.ds.repeat()
        self.ds = self.ds.prefetch(buffer_size=self.AUTOTUNE)

    def _encoder(self, x, filters):
        F1, F2, F3, F4, F5 = filters

        x = Conv2D(filters = F1,
            kernel_size = (1, 1),
            strides = (1,1), padding = 'valid',
            kernel_initializer = glorot_uniform(seed=0))(x)
        x = BatchNormalization(axis = 3)(x)
        x = Activation('relu')(x)

        x = MaxPooling2D((2,2), strides=(2,2))(x)

        x = Conv2D(filters = F2,
            kernel_size = (1, 1),
            strides = (1,1), padding = 'valid',
            kernel_initializer = glorot_uniform(seed=0))(x)
        x = BatchNormalization(axis = 3)(x)
        x = Activation('relu')(x)

        x = MaxPooling2D((2,2), strides=(2,2))(x)

        x = Conv2D(filters = F3,
            kernel_size = (1, 1),
            strides = (1,1), padding = 'valid',
            kernel_initializer = glorot_uniform(seed=0))(x)
        x = BatchNormalization(axis = 3)(x)
        x = Activation('relu')(x)

        x = MaxPooling2D((2,2), strides=(2,2))(x)

        x = Conv2D(filters = F4,
            kernel_size = (1, 1),
            strides = (1,1), padding = 'valid',
            kernel_initializer = glorot_uniform(seed=0))(x)
        x = BatchNormalization(axis = 3)(x)
        x = Activation('relu')(x)

        x = MaxPooling2D((2,2), strides=(2,2))(x)

        x = Conv2D(filters = F5,
            kernel_size = (1, 1),
            strides = (1,1), padding = 'valid',
            kernel_initializer = glorot_uniform(seed=0))(x)
        x = BatchNormalization(axis = 3)(x)
        x = Activation('relu')(x)

        return x

    def _decoder(self, x, filters):
        F1, F2, F3, F4, F5 = filters
        x = Conv2DTranspose(filters = F1,
            kernel_size = (1, 1),
            strides = (1,1), padding = 'valid',
            kernel_initializer = glorot_uniform(seed=0))(x)
        x = BatchNormalization(axis = 3)(x)
        x = Activation('relu')(x)

        x = UpSampling2D()(x)

        x = Conv2DTranspose(filters = F2,
            kernel_size = (1, 1),
            strides = (1,1), padding = 'valid',
            kernel_initializer = glorot_uniform(seed=0))(x)
        x = BatchNormalization(axis = 3)(x)
        x = Activation('relu')(x)

        x = UpSampling2D()(x)

        x = Conv2DTranspose(filters = F3,
            kernel_size = (1, 1),
            strides = (1,1), padding = 'valid',
            kernel_initializer = glorot_uniform(seed=0))(x)
        x = BatchNormalization(axis = 3)(x)
        x = Activation('relu')(x)

        x = UpSampling2D()(x)

        x = Conv2DTranspose(filters = F4,
            kernel_size = (1, 1),
            strides = (1,1), padding = 'valid',
            kernel_initializer = glorot_uniform(seed=0))(x)
        x = BatchNormalization(axis = 3)(x)
        x = Activation('relu')(x)

        x = UpSampling2D()(x)

        x = Conv2DTranspose(filters = F5,
            kernel_size = (1, 1),
            strides = (1,1), padding = 'valid',
            kernel_initializer = glorot_uniform(seed=0))(x)
        x = BatchNormalization(axis = 3)(x)
        x = Activation('relu')(x)

        return x


    def build(self, shape=None, load=None):
        self.shape = (self.img_rows,self.img_cols,self.channels)
        if shape is not None:
            self.shape = shape

        # build network
        input_a = Input(shape=self.shape)
        input_b = Input(shape=self.shape)
        input_c = Input(shape=self.shape)

        # encode

        a = self._encoder(input_a, [61,128,256,128,64])
        b = self._encoder(input_b, [64,128,256,128,64])
        c = self._encoder(input_c, [64,128,256,128,64])

        # decode
        a = self._decoder(a, [64,128,64,32,1])
        b = self._decoder(b, [64,128,64,32,1])
        c = self._decoder(c, [64,128,64,32,1])

        euclidian_ab = Euclidian(1)([a,b])
        euclidian_ac = Euclidian(1)([a,c])

        merged = tf.stack([euclidian_ab,euclidian_ac],axis=1)

        output = Softmax()(merged)

        self.nn = Model(inputs = [input_a,input_b,input_c], outputs = output)
        self.nn.compile(optimizer = self.optimizer,loss=self.loss, metrics=self.metrics)
        self.nn.summary()

        if load is not None:
            self.nn = load_model(self.save_path + "CNN.h5")

    def train(self, epoch, load=None):
        if load is not None:
            self.nn.load_weights(self.save_path)
        history = self.nn.fit(self.ds,epochs=epoch,steps_per_epoch=5000)
        self.nn.save(self.save_path+"CNN.h5")
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
