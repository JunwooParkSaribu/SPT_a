import os
import sys
import numpy as np
import tensorflow as tf
import json
from tensorflow import keras
from keras import layers

print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))


def shuffle(data, *args):
    shuffle_index = np.arange(data.shape[0])
    np.random.shuffle(shuffle_index)
    args = list(args)
    for i, arg in enumerate(args):
        args[i] = arg[shuffle_index]
    return data[shuffle_index], *args


loaded = np.load(f'./training_data/cls_training_set_20.npz')
input_signals_0 = loaded['input_signals_0']
input_labels_0 = loaded['input_labels_0']
input_signals_1 = loaded['input_signals_1']
input_labels_1 = loaded['input_labels_1']
val_input_signals_0 = loaded['val_input_signals_0']
val_input_labels_0 = loaded['val_input_labels_0']
val_input_signals_1 = loaded['val_input_signals_1']
val_input_labels_1 = loaded['val_input_labels_1']


train_input, train_label = shuffle(np.concatenate((input_signals_0, input_signals_1)), np.concatenate((input_labels_0, input_labels_1)))
val_input, val_label = shuffle(np.concatenate((val_input_signals_0, val_input_signals_1)), np.concatenate((val_input_labels_0, val_input_labels_1)))

train_input = train_input.reshape(-1, 20, 1, 4)
train_label = train_label.reshape(-1, 1)
val_input = val_input.reshape(-1, 20, 1, 4)
val_label = val_label.reshape(-1, 1)

print(train_input.shape, val_input.shape)


cls_input = keras.Input(shape=(None, 1, 4), name="reg_signals")
x = layers.ConvLSTM1D(filters=512, kernel_size=5, strides=1, return_sequences=True,
                          padding='same', dropout=0.1, data_format="channels_last")(cls_input)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM1D(filters=512, kernel_size=4, strides=1, return_sequences=True,
                          padding='same', dropout=0.1, data_format="channels_last")(x)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM1D(filters=512, kernel_size=3, strides=1, return_sequences=True,
                          padding='same', dropout=0.1, data_format="channels_last")(x)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM1D(filters=256, kernel_size=3, strides=1, return_sequences=True,
                          padding='same', dropout=0.1, data_format="channels_last")(x)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM1D(filters=256, kernel_size=2, strides=1, return_sequences=True,
                          padding='same', data_format="channels_last")(x)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM1D(filters=256, kernel_size=2, strides=1,
                          padding='same', data_format="channels_last")(x)
x = layers.BatchNormalization()(x)
x = layers.Flatten()(x)
x = layers.Dense(units=128, activation='leaky_relu')(x)
x = layers.Dense(units=64, activation='leaky_relu')(x)
cls_last_layer = layers.Dense(units=1, activation='sigmoid')(x)

cls_model = keras.Model(
        inputs=[cls_input],
        outputs=[cls_last_layer],
        name='anomalous_regression')


cls_model.compile(loss=tf.keras.losses.BinaryCrossentropy(name='binary_entropy_loss'),
                  optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3, momentum=0.1),
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='binary_acc'),
                           tf.keras.metrics.FalsePositives(name='FP'),
                           tf.keras.metrics.FalseNegatives(name='FN')
                          ])
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=25,
                                                  mode='min',
                                                  verbose=1,
                                                  restore_best_weights=True,
                                                  start_from_epoch=3)
cls_model.summary()


cls_history = cls_model.fit(x=train_input,
                            y=train_label,
                            validation_data=(val_input, val_label),
                            batch_size=64,
                            epochs=1500,
                            shuffle=True,
                            callbacks=[early_stopping],
                            verbose=2
                           )

cls_model.save(f'./models/cls_model_.keras')
history_dict = cls_history.history
json.dump(history_dict, open(f'./models/cls_history_.json', 'w'))
