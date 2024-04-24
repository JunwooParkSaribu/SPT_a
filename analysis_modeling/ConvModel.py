import tensorflow as tf
from keras.layers import Dense, Flatten, BatchNormalization, \
    Activation, Conv2D, AveragePooling2D, Dropout, ReLU, MaxPool2D
print("TensorFlow version:", tf.__version__)


class SPT(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layer1 = tf.keras.layers.Conv2D(512, kernel_size=(2, 2), padding='same')
        self.dense1 = Dense(10)
        self.drop = Dropout(0.1)
        self.flatten = Flatten()
        self.last_layer = Dense(units=1, activation='sigmoid')


    def call(self, x, training=None):
        x = self.layer1(x, training=training)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.drop(x)
        x = self.last_layer(x)
        return x

    """
    @tf.function
    def train_step(self, data):
        # Unpack the data
        x, y = data
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
        self.train_loss.update_state(loss)
        self.train_accuracy.update_state(y, y_pred)
        self.train_jsc.update_state(y, y_pred)

    @tf.function
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