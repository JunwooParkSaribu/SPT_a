import numpy as np
import tensorflow as tf
import json
from tensorflow import keras
from keras import layers
from andi_datasets.models_phenom import models_phenom


print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))


N = 3
D = 0.1
Ts = np.arange(8, 17, 4).astype(int)


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
    total_range = T + 200
    input_data = []
    input_label = []
    for i in range(12000):
        alpha = np.random.uniform(low=0.001, high=1.999)
        # alpha = np.random.choice([0.01, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 1.99], 1)[0]
        trajs_model, labels_model = models_phenom().single_state(N=N,
                                                                 L=None,
                                                                 T=int(total_range),
                                                                 alphas=alpha,  # Fixed alpha for each state
                                                                 Ds=[D, 0],  # Mean and variance of each state
                                                                 )
        for n_traj in range(N):
            # var_length = np.random.randint(-4, 4)
            xs = trajs_model[:, n_traj, 0][:T]
            ys = trajs_model[:, n_traj, 1][:T]
            xs = xs / (np.std(xs))
            xs = np.cumsum(abs(uncumulate(xs))) / T
            ys = ys / (np.std(ys))
            ys = np.cumsum(abs(uncumulate(ys))) / T
            input_data.append((xs + ys) / 2)
            input_label.append(alpha)

            for _ in range(25):
                # var_length = np.random.randint(-4, 4)
                random_start = np.random.randint(10, total_range - T)
                xs = trajs_model[:, n_traj, 0][random_start:random_start + T]
                ys = trajs_model[:, n_traj, 1][random_start:random_start + T]
                xs = xs / (np.std(xs))
                xs = np.cumsum(abs(uncumulate(xs))) / T
                ys = ys / (np.std(ys))
                ys = np.cumsum(abs(uncumulate(ys))) / T
                input_data.append((xs + ys) / 2)
                input_label.append(alpha)

    input_data = np.array(input_data).reshape(-1, 1, T, 1)
    input_label = np.array(input_label).reshape(-1, 1)
    input_data, input_label = shuffle(input_data, input_label)

    train_input = input_data[:int(input_data.shape[0] * 0.8)]
    train_label = input_label[:int(input_data.shape[0] * 0.8)]
    val_input = input_data[int(input_data.shape[0] * 0.8):]
    val_label = input_label[int(input_data.shape[0] * 0.8):]

    train_input, train_label = shuffle(train_input, train_label)
    val_input, val_label = shuffle(val_input, val_label)

    print(f'train_reg_shape:{train_input.shape}\n',
          f'train_label_shape:{train_label.shape}\n'
          f'val_reg_shape:{val_input.shape}\n',
          f'val_label_shape:{val_label.shape}\n'
         )

    # Shape [batch, time, features] => [batch, time, lstm_units]
    reg_input = keras.Input(shape=(1, None, 1), name="reg_signals")
    x = layers.ConvLSTM1D(filters=32, kernel_size=2, strides=1, padding='same', dropout=0.1)(reg_input)
    x = layers.ReLU()(x)
    x = layers.Bidirectional(layers.LSTM(32, dropout=0.1))(x)

    x = layers.Flatten()(x)
    reg_dense = layers.Dense(units=2, activation='relu')(x)
    reg_last_layer = layers.Dense(units=1)(reg_dense)

    reg_model = keras.Model(
        inputs=[reg_input],
        outputs=[reg_last_layer],
        name='anomalous_regression'
    )

    reg_model.compile(loss=tf.keras.losses.MeanSquaredError(name='mean_squared_error'),
                      optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3/2),
                      metrics=[tf.keras.metrics.MeanAbsoluteError(name='MAE'),
                              ]
                     )
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=50,
                                                      mode='min',
                                                      verbose=1,
                                                      restore_best_weights=True,
                                                      start_from_epoch=5
                                                     )

    reg_history = reg_model.fit(x=train_input,
                                y=train_label,
                                validation_data=(val_input, val_label),
                                batch_size=1024,
                                epochs=1500,
                                shuffle=True,
                                callbacks=[early_stopping],
                                verbose=2
                                )

    reg_model.save(f'./models/reg_model_{T}.keras')
    history_dict = reg_history.history
    json.dump(history_dict, open(f'./models/reg_history_{T}.json', 'w'))
