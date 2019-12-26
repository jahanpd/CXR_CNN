import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Add, Dense, Activation, Softmax, BatchNormalization, Flatten, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from algorithms.custom import Euclidian
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from pathlib import Path

# tf.enable_eager_execution()

class autoencoder:
    def __init__(self, save_path=None):
        K.set_floatx('float32')

        if save_path is not None:
            self.save_path = save_path

        self.img_rows = 256
        self.img_cols = 256
        self.channels = 1

        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.optimizer = 'Adam'
        self.loss = 'binary_crossentropy'
        self.metrics = [tf.keras.metrics.BinaryAccuracy()]

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

    def _input_pred(self, x_a,x_b,x_c):
        a = tf.data.Dataset.from_tensor_slices((x_a))
        a = a.map(self._load_and_preprocess_image, num_parallel_calls=self.AUTOTUNE)
        b = tf.data.Dataset.from_tensor_slices((x_b))
        b = b.map(self._load_and_preprocess_image, num_parallel_calls=self.AUTOTUNE)
        c = tf.data.Dataset.from_tensor_slices((x_c))
        c = c.map(self._load_and_preprocess_image, num_parallel_calls=self.AUTOTUNE)
        dataset = tf.data.Dataset.zip(({"input_1": a, "input_2": b, "input_3": c}))
        dataset = dataset.batch(10).repeat()
        return dataset

    def _input_valid(self, x_a,x_b,x_c, y):
        a = tf.data.Dataset.from_tensor_slices((x_a))
        a = a.map(self._load_and_preprocess_image, num_parallel_calls=self.AUTOTUNE)
        b = tf.data.Dataset.from_tensor_slices((x_b))
        b = b.map(self._load_and_preprocess_image, num_parallel_calls=self.AUTOTUNE)
        c = tf.data.Dataset.from_tensor_slices((x_c))
        c = c.map(self._load_and_preprocess_image, num_parallel_calls=self.AUTOTUNE)
        label =  tf.data.Dataset.from_tensor_slices((y))
        dataset = tf.data.Dataset.zip(({"input_1": a, "input_2": b, "input_3": c}, label))
        dataset = dataset.batch(10).repeat()
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
            self.nn.load_weights(self.save_path + load)

    def train(self, a_paths, b_paths, c_paths, epochs=10, load=None):

        y = np.full((len(a_paths),2),[0,1])
        X = np.dstack((a_paths, b_paths, c_paths))[0]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1000, random_state=42)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(self.save_path+"/triple.ckpt",
                                                 save_weights_only=True,
                                                 verbose=1)

        es = EarlyStopping(monitor='val_loss', patience = 5, verbose=1,restore_best_weights=True)
        if load is not None:
            self.nn.load_weights(self.save_path + load)

        history = self.nn.fit(self._input_fn(X_train[:,0],X_train[:,1],X_train[:,2],y_train),
                              epochs=epochs,
                              steps_per_epoch=4*100000,
                              callbacks = [cp_callback, es],
                              validation_data=self._input_valid(X_test[:,0],X_test[:,1],X_test[:,2],y_test),
                              validation_steps=100)

        self.nn.save_weights(self.save_path + "triple.weights")
        return history

    def predict(self,x_a,x_b,x_v, y):
        predictions = self.nn.predict(self._input_fn(x_a,x_b,x_v, y))
        return [self.nn.evaluate(self._input_fn(x_a,x_b,x_v, y)),predictions]

    def predict_proba(self, x_a,x_b,x_c, y):
        predictions = np.array(()) # array to store predictions
        idx = np.random.choice(len(y), 100,  replace=True)
        temp_b,temp_c = x_b[idx], x_c[idx]
        for n in range(len(y)):
            temp = np.array(()) # array to store predictions
            x_same = np.tile(x_a[n],100)
            prediction = self.nn.predict(self._input_pred(x_same,temp_b,temp_c), steps=10)
            temp = np.mean(prediction, axis=0)
            predictions = np.append(predictions, 1-temp[0])
            prob_aki = np.array([x if y == 1 else 1-x for x,y in zip(predictions.flatten(), y[idx])])
        return prob_aki
