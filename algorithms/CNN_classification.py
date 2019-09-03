import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, UpSampling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
import tensorflow.keras.backend as K
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from pathlib import Path

tf.enable_eager_execution()

class convNN:
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

    def identity_block(self,f, k, filters, stage, block):
        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        F1, F2, F3 = filters
        f_shortcut = f

        # first convolutional layer
        f = Conv2D(filters = F1,
                    kernel_size = (1, 1),
                    strides = (1,1), padding = 'valid',
                    name = conv_name_base + '2a',
                    kernel_initializer = glorot_uniform(seed=0))(f)
        f = BatchNormalization(axis = 3, name = bn_name_base + '2a')(f)
        f = Activation('relu')(f)

        # second convolutional layer
        f = Conv2D(filters = F2,
                    kernel_size = (k, k),
                    strides = (1,1), padding = 'same',
                    name = conv_name_base + '2b',
                    kernel_initializer = glorot_uniform(seed=0))(f)
        f = BatchNormalization(axis = 3, name = bn_name_base + '2b')(f)
        f = Activation('relu')(f)

        #third convolutional layer
        f = Conv2D(filters = F3,
                    kernel_size = (1, 1),
                    strides = (1,1), padding = 'valid',
                    name = conv_name_base + '2c',
                    kernel_initializer = glorot_uniform(seed=0))(f)
        f = BatchNormalization(axis = 3, name = bn_name_base + '2c')(f)
        f = Activation('relu')(f)

        # add shortcut and generate residual
        f = Add()([f, f_shortcut])
        f = Activation('relu')(f)

        return f

    def convolutional_block(self,f, k, filters, stage, block, s=2):
        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        F1, F2, F3 = filters
        f_shortcut = f

        # first convolutional layer
        f = Conv2D(filters = F1,
            kernel_size = (1, 1),
            strides = (s,s), padding = 'valid',
            name = conv_name_base + '2a',
            kernel_initializer = glorot_uniform(seed=0))(f)
        f = BatchNormalization(axis = 3, name = bn_name_base + '2a')(f)
        f = Activation('relu')(f)

        # second convolutional layer
        f = Conv2D(filters = F2,
            kernel_size = (k, k),
            strides = (1,1), padding = 'same',
            name = conv_name_base + '2b',
            kernel_initializer = glorot_uniform(seed=0))(f)
        f = BatchNormalization(axis = 3, name = bn_name_base + '2b')(f)
        f = Activation('relu')(f)

        #third convolutional layer
        f = Conv2D(filters = F3,
            kernel_size = (1, 1),
            strides = (1,1), padding = 'valid',
            name = conv_name_base + '2c',
            kernel_initializer = glorot_uniform(seed=0))(f)
        f = BatchNormalization(axis = 3, name = bn_name_base + '2c')(f)
        f = Activation('relu')(f)

        # add shortcut layer
        f_shortcut = Conv2D(filters = F3,
                        kernel_size = (1, 1),
                        strides = (s,s), padding = 'valid',
                        name = conv_name_base + '1',
                        kernel_initializer = glorot_uniform(seed=0))(f_shortcut)
        f_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(f_shortcut)


        f = Add()([f, f_shortcut])
        f = Activation('relu')(f)

        return f


    def build(self, shape=None, load=None):
        self.shape = (self.img_rows,self.img_cols,self.channels)
        if shape is not None:
            self.shape = shape

        # build network
        input = Input(shape=self.shape)

        f = Conv2D(filters = 16,
            kernel_size = (5, 5),
            strides = (1,1), padding = 'same',
            name = 'conv1',
            kernel_initializer = glorot_uniform(seed=0))(input)
        f = BatchNormalization(axis = 3, name = '12c')(f)
        f = Activation('relu')(f)

        f = self.convolutional_block(f, k=3, filters=[16,16,32], stage='2', block='a', s=2)
        f = self.identity_block(f, k=3, filters=[32,32,32], stage='2', block='b')
        f = self.identity_block(f, k=3, filters=[32,32,32], stage='2', block='c')
        f = MaxPooling2D((3,3),strides=(2,2))(f)

        f = self.convolutional_block(f, k=3, filters=[32,32,64], stage='3', block='a', s=2)
        f = self.identity_block(f, k=3, filters=[64,64,64], stage='3', block='b')
        f = self.identity_block(f, k=3, filters=[64,64,64], stage='3', block='c')

        f = self.convolutional_block(f, k=3, filters=[32,32,64], stage='4', block='a', s=2)
        f = self.identity_block(f, k=3, filters=[64,64,64], stage='4', block='b')
        f = self.identity_block(f, k=3, filters=[64,64,64], stage='4', block='c')

        f = GlobalAveragePooling2D()(f)

        output = Dense(2,activation="softmax")(f)

        self.nn = Model(inputs = input, outputs = output)
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
