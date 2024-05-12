import os
import sys
import json
sys.path.insert(0, os.path.abspath('../'))
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from keras import layers

print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))

SHIFT_WIDTH = 40
REG_JUMP = 2

SHUFFLE = True
MAX_EPOCHS = 10000
BATCH_SIZE = 1024
PATIENCE = 500
NB_FEATURES = 2

loaded = np.load(f'./training_data/training_set_{SHIFT_WIDTH}_{REG_JUMP}.npz')
input_signals = loaded['input_signals']
input_labels = loaded['input_labels']
input_features = loaded['input_features']
input_reg_signals = loaded['input_reg_signals']
input_reg_labels = loaded['input_reg_labels']
count_0 = loaded['count_0']
count_1 = loaded['count_1']

total = count_0 + count_1
weight_for_0 = (1 / count_0) * (total / 2.0)
weight_for_1 = (1 / count_1) * (total / 2.0)
class_weight = {0: weight_for_0, 1:weight_for_1}

print(input_signals.shape, input_labels.shape)
print(input_reg_signals.shape, input_reg_labels.shape)
print(class_weight, count_0, count_1)
print(input_features.shape)


def shuffle(data, *args):
    shuffle_index = np.arange(data.shape[0])
    np.random.shuffle(shuffle_index)
    args = list(args)
    for i, arg in enumerate(args):
        args[i] = arg[shuffle_index]
    return data[shuffle_index], *args



input_signals = input_signals[:, :, input_signals.shape[-1]//2:]
input_reg_signals = input_reg_signals[:, :, input_reg_signals.shape[-1]//2:]
#input_signals = np.swapaxes(input_signals, 1, 2)
#input_reg_signals = np.swapaxes(input_reg_signals, 1, 2)
INPUT_CLS_SHAPE = [-1, input_signals.shape[1], input_signals.shape[2], 1]
INPUT_REG_SHAPE = [-1, input_reg_signals.shape[1], input_reg_signals.shape[2], 1]


input_signals = input_signals.reshape(INPUT_CLS_SHAPE)
input_labels = input_labels.reshape(-1, 1)
input_reg_signals = input_reg_signals.reshape(INPUT_REG_SHAPE)
input_reg_labels = input_reg_labels.reshape(-1, 1)
input_signals, input_labels, input_features = shuffle(input_signals, input_labels, input_features)
input_reg_signals, input_reg_labels = shuffle(input_reg_signals, input_reg_labels)

train_input = []
train_label = []
train_feature = []
val_input = []
val_label = []
val_feature = []
cur_count_0 = 0
cur_count_1 = 0
for i in range(len(input_labels)):
    if input_labels[i] == 0:
        cur_count_0 += 1
        if cur_count_0 < int(count_0 * 0.8):
            train_input.append(input_signals[i])
            train_label.append(input_labels[i])
            train_feature.append(input_features[i])
        else:
            val_input.append(input_signals[i])
            val_label.append(input_labels[i])
            val_feature.append(input_features[i])
    else:
        cur_count_1 += 1
        if cur_count_1 < int(count_1 * 0.8):
            train_input.append(input_signals[i])
            train_label.append(input_labels[i])
            train_feature.append(input_features[i])
        else:
            val_input.append(input_signals[i])
            val_label.append(input_labels[i])
            val_feature.append(input_features[i])

train_input = np.array(train_input)
train_label = np.array(train_label)
train_feature = np.array(train_feature)
val_input = np.array(val_input)
val_label = np.array(val_label)
val_feature = np.array(val_feature)
train_reg_input = input_reg_signals[:int(input_reg_signals.shape[0] * 0.8)]
train_reg_label = input_reg_labels[:int(input_reg_labels.shape[0] * 0.8)]
val_reg_input = input_reg_signals[int(input_reg_signals.shape[0] * 0.8):]
val_reg_label = input_reg_labels[int(input_reg_labels.shape[0] * 0.8):]


train_input, train_label, train_feature = shuffle(train_input, train_label, train_feature)
val_input, val_label, val_feature = shuffle(val_input, val_label, val_feature)
train_reg_input, train_reg_label = shuffle(train_reg_input, train_reg_label)
val_reg_input, val_reg_label = shuffle(val_reg_input, val_reg_label)


print(f'train_cls_shape:{train_input.shape}\n',
      f'train_feat_shape:{train_feature.shape}\n'
      f'val_cls_shape:{val_input.shape}\n',
      f'val_feat_shape:{val_feature.shape}\n'
      f'train_reg_shape:{train_reg_input.shape}\n',
      f'val_reg_shape:{val_reg_input.shape}\n',
     )



"""
signal_input = keras.Input(shape=train_input.shape[1:], name="signals")
feature_input = keras.Input(shape=train_feature.shape[1:], name="features")

x1 = layers.ConvLSTM1D(filters=256, kernel_size=2, strides=1, padding='same', dropout=0.1)(signal_input)
x1 = layers.ReLU()(x1)
x1 = layers.Bidirectional(layers.LSTM(256))(x1)
x1 = layers.ReLU()(x1)
x1 = layers.Flatten()(x1)

x2 = layers.Dense(units=train_input.shape[-1])(feature_input)
x2 = layers.ReLU()(x2)
x2 = layers.Dense(units=train_input.shape[-1] * 2)(x2)
x2 = layers.ReLU()(x2)
x2 = layers.Flatten()(x2)
cls_concat = layers.concatenate([x1, x2])
cls_dense = layers.Dense(units=64, activation='relu')(cls_concat)
cls_last_layer = layers.Dense(units=1, activation='sigmoid')(cls_dense)

cls_model = keras.Model(
    inputs=[signal_input, feature_input],
    outputs=[cls_last_layer],
    name='anomalous_detection'
)


cls_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='Acc'),
                           tf.keras.metrics.FalsePositives(name='FP'),
                           tf.keras.metrics.FalseNegatives(name='FN')]
                 )
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_Acc',
                                                  patience=PATIENCE,
                                                  mode='min',
                                                  verbose=1,
                                                  restore_best_weights=True,
                                                  start_from_epoch=15
                                                 )
try:
    cls_model.summary()
    keras.utils.plot_model(cls_model, "cls_model.png", show_shapes=True)
except:
    pass

cls_history = cls_model.fit(x=[train_input, train_feature],
                        y=train_label,
                        validation_data=([val_input, val_feature], val_label),
                        batch_size=BATCH_SIZE,
                        epochs=MAX_EPOCHS,
                        shuffle=True,
                        callbacks=[early_stopping],
                        class_weight=class_weight,
                        verbose=2
                       )

cls_model.save(f'./models/cls_model_{SHIFT_WIDTH}_{REG_JUMP}.keras')
history_dict = cls_history.history
json.dump(history_dict, open(f'./models/history_{SHIFT_WIDTH}_{REG_JUMP}.json', 'w'))

del cls_model
del history_dict
del train_input
del train_label
del train_feature
del val_input
del val_label
del val_feature
del input_signals
del input_labels
del input_features
"""

############# REGRESSION ###############


reg_input = keras.Input(shape=train_reg_input.shape[1:], name="reg_signals")

x = layers.ConvLSTM1D(filters=256, kernel_size=2, strides=1, padding='same', dropout=0.1)(reg_input)
x = layers.ReLU()(x)
x = layers.Bidirectional(layers.LSTM(256))(x)
x = layers.ReLU()(x)
x = layers.Flatten()(x)
reg_dense = layers.Dense(units=64, activation='relu')(x)
reg_last_layer = layers.Dense(units=32)(reg_dense)

reg_model = keras.Model(
    inputs=[reg_input],
    outputs=[reg_last_layer],
    name='anomalous_regression'
)

reg_model.compile(loss=tf.keras.losses.MeanSquaredError(name='mean_squared_error'),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  metrics=[tf.keras.metrics.MeanAbsoluteError(name='MAE'),
                          ]
                 )
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=PATIENCE,
                                                  mode='min',
                                                  verbose=1,
                                                  restore_best_weights=True,
                                                  start_from_epoch=15
                                                 )

reg_history = reg_model.fit(x=train_reg_input,
                        y=train_reg_label,
                        validation_data=(val_reg_input, val_reg_label),
                        batch_size=BATCH_SIZE,
                        epochs=MAX_EPOCHS,
                        shuffle=True,
                        callbacks=[early_stopping],
                        verbose=2
                       )
reg_model.save(f'./models/reg_model_{SHIFT_WIDTH}_{REG_JUMP}.keras')
history_dict = reg_history.history
json.dump(history_dict, open(f'./models/reg_history_{SHIFT_WIDTH}_{REG_JUMP}.json', 'w'))