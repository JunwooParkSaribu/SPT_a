import numpy as np
import tensorflow as tf
import json
from tensorflow import keras
from keras import layers
from andi_datasets.models_phenom import models_phenom


print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))


N = 10
Ts = [32, 16]


def radius_list(xs:np.ndarray, ys:np.ndarray):
    assert xs.ndim == 1 and ys.ndim == 1
    rad_list = [0.]
    disp_list = []
    for i in range(1, len(xs)):
        rad_list.append(np.sqrt((xs[i] - xs[0])**2 + (ys[i] - ys[0])**2))
        disp_list.append(np.sqrt((xs[i] - xs[i-1])**2 + (ys[i] - ys[i-1])**2))
    return np.array(rad_list) / np.mean(disp_list) / len(xs)


def uncumulate(xs:np.ndarray):
    assert xs.ndim == 1
    uncum_list = [0.]
    for i in range(1, len(xs)):
        uncum_list.append(xs[i] - xs[i-1])
    return np.array(uncum_list).copy()


def shuffle(data, *args):
    shuffle_index = np.arange(data.shape[0])
    np.random.shuffle(shuffle_index)
    args = list(args)
    for i, arg in enumerate(args):
        args[i] = arg[shuffle_index]
    return data[shuffle_index], *args


for T in Ts:
    print(f'--------------------------- {T} start ------------------------------')
    total_range = T + 200

    input_data = []
    input_label = []

    for alpha in np.arange(0.001, 2, 0.001):
        D = np.random.uniform(low=0.01, high=10.0)
        trajs_model, labels_model = models_phenom().single_state(N=N,
                                                                 L=None,
                                                                 T=total_range,
                                                                 alphas=alpha,
                                                                 Ds=[D, 0],
                                                                 )
        for n_traj in range(N):
            # var_length = np.random.randint(-4, 4)
            xs = trajs_model[:, n_traj, 0][:T]
            ys = trajs_model[:, n_traj, 1][:T]
            rad_list = radius_list(xs, ys)

            xs = xs / (np.std(xs))
            xs = np.cumsum(abs(uncumulate(xs))) / T
            ys = ys / (np.std(ys))
            ys = np.cumsum(abs(uncumulate(ys))) / T

            input_list = np.vstack((xs, rad_list)).T
            input_data.append(input_list)
            input_label.append(alpha)

            input_list = np.vstack((ys, rad_list)).T
            input_data.append(input_list)
            input_label.append(alpha)

            for random_start in range(5, total_range - T, 3):
                # var_length = np.random.randint(-4, 4)
                # random_start = np.random.randint(5, total_range - T)
                xs = trajs_model[:, n_traj, 0][random_start:random_start + T]
                ys = trajs_model[:, n_traj, 1][random_start:random_start + T]
                rad_list = radius_list(xs, ys)

                xs = xs / (np.std(xs))
                xs = np.cumsum(abs(uncumulate(xs))) / T
                ys = ys / (np.std(ys))
                ys = np.cumsum(abs(uncumulate(ys))) / T

                input_list = np.vstack((xs, rad_list)).T
                input_data.append(input_list)
                input_label.append(alpha)

                input_list = np.vstack((ys, rad_list)).T
                input_data.append(input_list)
                input_label.append(alpha)

    input_data = np.array(input_data)
    input_label = np.array(input_label)

    valid_data = []
    valid_label = []

    for alpha in np.arange(0.001, 2, 0.001):
        D = np.random.uniform(low=0.01, high=10.0)
        trajs_model, labels_model = models_phenom().single_state(N=3,
                                                                 L=None,
                                                                 T=total_range,
                                                                 alphas=alpha,
                                                                 Ds=[D, 0],
                                                                 )
        for n_traj in range(3):
            for random_start in range(5, total_range - T, 3):
                #random_start = np.random.randint(0, total_range - T)
                xs = trajs_model[:, n_traj, 0][random_start:random_start + T]
                ys = trajs_model[:, n_traj, 1][random_start:random_start + T]
                rad_list = radius_list(xs, ys)

                xs = xs / (np.std(xs))
                xs = np.cumsum(abs(uncumulate(xs))) / T
                ys = ys / (np.std(ys))
                ys = np.cumsum(abs(uncumulate(ys))) / T

                input_list = np.vstack((xs, rad_list)).T
                valid_data.append(input_list)
                valid_label.append(alpha)

                input_list = np.vstack((ys, rad_list)).T
                valid_data.append(input_list)
                valid_label.append(alpha)

    valid_data = np.array(valid_data)
    valid_label = np.array(valid_label)

    train_input, train_label = shuffle(input_data, input_label)
    val_input, val_label = shuffle(valid_data, valid_label)

    train_input = train_input.reshape(-1, T, 1, 2)
    train_label = train_label.reshape(-1, 1)
    val_input = val_input.reshape(-1, T, 1, 2)
    val_label = val_label.reshape(-1, 1)

    print(f'train_reg_shape:{train_input.shape}\n',
          f'train_label_shape:{train_label.shape}\n'
          f'val_reg_shape:{val_input.shape}\n',
          f'val_label_shape:{val_label.shape}\n'
         )

    # Shape [batch, time, features] => [batch, time, lstm_units]
    reg_input = keras.Input(shape=(None, 1, 2), name="reg_signals")
    x = layers.ConvLSTM1D(filters=512, kernel_size=3, strides=1, return_sequences=True,
                          padding='same', dropout=0.1, data_format="channels_last")(reg_input)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM1D(filters=256, kernel_size=3, strides=1, return_sequences=True,
                          padding='same', dropout=0.1, data_format="channels_last")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM1D(filters=256, kernel_size=3, strides=1, return_sequences=True,
                          padding='same', dropout=0.1, data_format="channels_last")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM1D(filters=128, kernel_size=3, strides=1, return_sequences=True,
                          padding='same', data_format="channels_last")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM1D(filters=128, kernel_size=3, strides=1, return_sequences=True,
                          padding='same', data_format="channels_last")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(units=64, activation='leaky_relu')(x)
    reg_last_layer = layers.Dense(units=1, activation='relu')(x)

    reg_model = keras.Model(
        inputs=[reg_input],
        outputs=[reg_last_layer],
        name='anomalous_regression'
    )

    reg_model.compile(loss=tf.keras.losses.Huber(name='huber_loss'),
                      optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3, momentum=0.1),
                      metrics=[tf.keras.metrics.MeanAbsoluteError(name='MAE'),
                               ]
                      )
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=20,
                                                      mode='min',
                                                      verbose=1,
                                                      restore_best_weights=True,
                                                      start_from_epoch=3
                                                      )

    reg_history = reg_model.fit(x=train_input,
                                y=train_label,
                                validation_data=(val_input, val_label),
                                batch_size=2048,
                                epochs=1500,
                                shuffle=True,
                                callbacks=[early_stopping],
                                verbose=2
                                )
    reg_model.save(f'./models/reg_model_{T}.keras')
    history_dict = reg_history.history
    json.dump(history_dict, open(f'./models/reg_history_{T}.json', 'w'))
    print(f'--------------------------- {T} finished ------------------------------')
