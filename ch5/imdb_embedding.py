
from keras import preprocessing
from keras.layers import Dense, Embedding, Flatten
from keras.models import Sequential
from keras.datasets import imdb


max_features = 10000
max_len = 20

(x_train, y_train), (x_test, y_test) = \
    imdb.load_data(num_words=max_features)
x_train = preprocessing.sequence.pad_sequences(
    x_train, maxlen=max_len)
x_test = preprocessing.sequence.pad_sequences(
    x_test, maxlen=max_len)

model = Sequential()
model.add(Embedding(max_features, 8, input_length=max_len))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', metrics=['acc'],
              loss='binary_crossentropy')
model.summary()

history = model.fit(x_train, y_train, epochs=10,
                    batch_size=32, validation_split=0.2)
