import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, UpSampling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform
import tensorflow.keras.backend as K
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from pathlib import Path

class convAutoencoder:
    def __init__(self, save_path=None, data_path=None):
        K.set_floatx('float16')
        if save_path is not None:
            self.save_path = save_path
        if data_path is not None:
            self.data = data_path

        if data_path is None:
            print("need to add location of data")

        self.img_rows = 256
        self.img_cols = 256
        self.channels = 1
        self.output = 256*256
        self.cluster = 1

    def _prep_extraction(self):
        self.filenames = []
        for filename in Path(self.data).glob('**/*.npz'):
                self.filenames.append(str(filename))
        self.indexes =(np.arange(len(self.filenames))).reshape((-1,self.cluster))
        np.random.shuffle(self.indexes)

    def _batch_data(self, iteration):
        x = np.array(())
        for n in range(self.cluster):
            data = np.load(self.filenames[self.indexes[iteration][n]])
            temp_x = data['a']
            _, heightx, widthx = temp_x.shape
            x = np.append(x, temp_x)
        return x.reshape((-1,heightx,widthx,1))/255

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


    def build(self, shape=None):

        self.shape = (256,256,1)
        if shape is not None:
            self.shape = shape

        # build network
        inputs = Input(shape=self.shape)

        ## ENCODE
        f = Conv2D(filters = 64,
            kernel_size = (5, 5),
            strides = (1,1), padding = 'same',
            name = 'conv1',
            kernel_initializer = glorot_uniform(seed=0))(inputs)
        f = BatchNormalization(axis = 3, name = '12c')(f)
        f = Activation('relu')(f)
        f = MaxPooling2D((3,3),strides=(2,2))(f)

        f = Conv2D(filters = 128,
            kernel_size = (5, 5),
            strides = (1,1), padding = 'same',
            name = 'conv2',
            kernel_initializer = glorot_uniform(seed=0))(f)
        f = BatchNormalization(axis = 3, name = '22c')(f)
        f = Activation('relu')(f)
        f = MaxPooling2D((3,3),strides=(2,2))(f)

        f = Conv2D(filters = 256,
            kernel_size = (5, 5),
            strides = (1,1), padding = 'same',
            name = 'conv3',
            kernel_initializer = glorot_uniform(seed=0))(f)
        f = BatchNormalization(axis = 3, name = '32c')(f)
        f = Activation('relu')(f)
        f = MaxPooling2D((3,3),strides=(2,2))(f)

        f = Conv2D(filters = 512,
            kernel_size = (5, 5),
            strides = (1,1), padding = 'same',
            name = 'conv4',
            kernel_initializer = glorot_uniform(seed=0))(f)
        f = BatchNormalization(axis = 3, name = '42c')(f)
        f = Activation('relu')(f)
        f = MaxPooling2D((3,3),strides=(2,2))(f)

        #f = GlobalAveragePooling2D()(f)

        ## DECODE
        # Stage 4 (≈6 lines)
        f = Conv2D(512, (4, 4), strides=(2, 2), name='conv5', padding='same', kernel_initializer=glorot_uniform(seed=0))(f)
        f = BatchNormalization(axis=3, name='bn_conv0')(f)
        f = Activation('relu')(f)
        f = UpSampling2D((2, 2))(f)

        f = Conv2D(256, (4, 4), strides=(1, 1), name='conv6', padding='same', kernel_initializer=glorot_uniform(seed=0))(f)
        f = BatchNormalization(axis=3, name='bn_conv')(f)
        f = Activation('relu')(f)
        f = UpSampling2D((2, 2))(f)

        f = Conv2D(128, (1, 1), activation='sigmoid', padding='same', name='fc1024', kernel_initializer = glorot_uniform(seed=0))(f)
        output = Flatten()(f)

        self.autoencoder = Model(inputs = inputs, outputs=output, name='ResNetAE')
        self.autoencoder.compile(optimizer='Adagrad', loss='mae')
        self.autoencoder.summary()

    def train(self, epochs, load=None):
        # prep data extraction file locations
        self._prep_extraction()
        batch_n, _ = self.indexes.shape

        if load is not None:
            self.autoencoder.load_weights(self.save_path)

        for yolo in range(epochs):
            for batch in range(batch_n):
                x_train = self._batch_data(batch)
                self.autoencoder.fit(x_train,x_train.reshape((-1,256*256)),epochs=100,batch_size=32)
                self.autoencoder.save_weights(self.save_path)
