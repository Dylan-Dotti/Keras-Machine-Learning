
from keras import models, layers
from keras.datasets import boston_housing
import matplotlib.pyplot as plt
import numpy as np


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


def k_fold(k=4, num_epochs=100):
    num_val_samples = len(train_data) // k
    all_mae_histories = []
    for i in range(k):
        print('processing fold #', i)
        val_data = train_data[i * num_val_samples:
                              (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples:
                                    (i + 1) * num_val_samples]
        partial_train_data = np.concatenate(
            [train_data[:i * num_val_samples],
             train_data[(i + 1) * num_val_samples:]],
            axis=0)
        partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples],
             train_targets[(i + 1) * num_val_samples:]],
            axis=0)
        model = build_model()
        history = model.fit(partial_train_data, partial_train_targets,
                            epochs=num_epochs, batch_size=1,
                            validation_data=(val_data, val_targets))
        mae_history = history.history['val_mean_absolute_error']
        all_mae_histories.append(mae_history)
    return all_mae_histories


print 'Loading data...'
(train_data, train_targets), (test_data, test_targets) = \
    boston_housing.load_data()
print 'Load complete'

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

model = build_model()
model.fit(train_data, train_targets,
          epochs=80, batch_size=16)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print test_mae_score
