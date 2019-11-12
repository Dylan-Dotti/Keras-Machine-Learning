
from keras import models, layers
from keras.datasets import imdb
import numpy as np
import matplotlib.pyplot as plt
import metric_plots


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


def encode_review(review):
    words = review.split()
    word_index = imdb.get_word_index()
    encoded = np.array([word_index[word] for word in words])
    return encoded


def decode_review(word_indexes):
    word_index = imdb.get_word_index()
    reverse_word_index = dict(
        [(value, key) for (key, value) in word_index.items()])
    decoded_review = ' '.join(
        [reverse_word_index.get(i - 3, '?') for i in word_indexes])
    return decoded_review


print 'Loading data...'
(train_data, train_labels), (test_data, test_labels) = \
    imdb.load_data(num_words=10000)
print 'Load complete'

print 'Vectorizing data...'
x_train = vectorize_sequences(train_data)
y_train = np.asarray(train_labels).astype('float32')
x_test = vectorize_sequences(test_data)
y_test = np.asarray(test_labels).astype('float32')
print 'Vectorization complete'

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

model = models.Sequential([
    layers.Dense(16, activation='relu', input_shape=(10000,)),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(x_train,
                    y_train,
                    epochs=4,
                    batch_size=256)
#plot_loss(history)
#plot_acc(history)

results = model.evaluate(x_test, y_test)
print('Test accuracy:', results[1])
