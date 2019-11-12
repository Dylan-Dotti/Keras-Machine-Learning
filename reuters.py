
from keras import models, layers
from keras.utils.np_utils import to_categorical
from keras.datasets import reuters
import numpy as np
import metric_plots as mplots


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


def decode_newswire(newswire):
    word_index = reuters.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in
                               word_index.items()])
    decoded = ' '.join([reverse_word_index.get(i - 3, '?')
                        for i in newswire])
    return decoded


print 'Loading data...'
(train_data, train_labels), (test_data, test_labels) = \
    reuters.load_data(num_words=10000)
print 'Load complete'

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]

model = models.Sequential()
model.add(layers.Dense(96, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(96, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(partial_x_train, partial_y_train,
                    epochs=9, batch_size=128,
                    validation_data=(x_val, y_val))

results = model.evaluate(x_test, y_test)
print results[1]

#metric_plots.plot_loss(history)
#mplots.plot_acc(history)