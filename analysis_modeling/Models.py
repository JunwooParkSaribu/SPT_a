import tensorflow as tf
print("TensorFlow version:", tf.__version__)


class Anomalous_detection(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #self.layer0 = tf.keras.layers.Conv2D(2, kernel_size=(5, 2), padding='same', strides=(5, 1))
        #self.relu0 = tf.keras.layers.ReLU()
        #self.dense0 = tf.keras.layers.Dense(1)
        #self.lstm = tf.keras.layers.LSTM(32)
        self.conv_lstm = tf.keras.layers.ConvLSTM1D(filters=32, kernel_size=2, strides=2, padding='same', dropout=0.1)
        self.bidirec_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))
        #self.drop = tf.keras.layers.Dropout(0.1)
        self.flatten = tf.keras.layers.Flatten()
        #self.dense0 = tf.keras.layers.Dense(16)
        self.last_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')


    def call(self, inputs, training=None):
        #x = self.layer0(inputs)
        #x = self.layer1(x)
        #x = self.relu0(x)
        #x = self.dense0(inputs)
        #x = self.drop(x, training=training)
        #x = self.lstm(inputs)
        x = self.conv_lstm(inputs)
        x = self.bidirec_lstm(x)
        #x = self.drop(x, training=training)
        x = self.flatten(x)
        #x = self.dense0(x)
        x = self.last_layer(x)
        return x


    """
    def train_step(self, data):
        print(self.metrics)
        # Unpack the data
        x = data
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            y_pred = self(x, training=True)
            loss = self.loss_object(y_true=y, y_pred=y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute the metrics
        self.loss.update_state(loss)
        for metric in self.metrics:
            tf.print(metric)
            metric.update_state(y, y_pred)
        self.train_accuracy.update_state(y, y_pred)
        self.train_jsc.update_state(y, y_pred)


    def test_step(self, data):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self(x, training=False)
        # Tracking the loss
        t_loss = self.loss_object(y, y_pred)
        # Update the metrics
        self.test_loss.update_state(t_loss)
        self.test_accuracy.update_state(y, y_pred)
        self.test_jsc.update_state(y, y_pred)
    """


class Anomalous_regression(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv_lstm = tf.keras.layers.ConvLSTM1D(filters=32, kernel_size=2, strides=2, padding='same', dropout=0.1)
        #self.drop = tf.keras.layers.Dropout(0.1)
        self.flatten = tf.keras.layers.Flatten()
        self.last_layer = tf.keras.layers.Dense(units=1)


    def call(self, inputs, training=None):
        x = self.conv_lstm(inputs)
        #x = self.drop(x, training=training)
        x = self.flatten(x)
        x = self.last_layer(x)
        return x