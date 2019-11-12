
import sys
import os
import numpy as np
import metric_plots
from keras import layers, models
from keras.applications import VGG16
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator


def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=RMSprop(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['acc'])
    return model


def build_conv_base_model():
    conv_base = VGG16(weights='imagenet', include_top=False,
                      input_shape=(150, 150, 3))
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    conv_base.trainable = False
    return model


def build_generators(train_dir, validation_dir):
    train_datagen = ImageDataGenerator(
        rescale=1./255, rotation_range=40,
        width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(150, 150),
        batch_size=32, class_mode='binary')
    validation_generator = test_datagen.flow_from_directory(
        validation_dir, target_size=(150, 150),
        batch_size=32, class_mode='binary')
    return train_generator, validation_generator


def build_generators_conv_base(train_dir, validation_dir):
    train_datagen = ImageDataGenerator(
        rescale=1./255, rotation_range=40,
        width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode='nearest')
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(150, 150),
        batch_size=20, class_mode='binary')
    validation_generator = test_datagen.flow_from_directory(
        validation_dir, target_size=(150, 150),
        batch_size=20, class_mode='binary')

    return train_generator, validation_generator


def extract_features(directory, sample_count):
    conv_base = VGG16(weights='imagenet', include_top=False,
                      input_shape=(150, 150, 3))
    data_gen = ImageDataGenerator(rescale=1./255)
    batch_size = 20

    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = data_gen.flow_from_directory(
        directory, target_size=(150, 150),
        batch_size=batch_size, class_mode='binary')
    for i, (inputs_batch, labels_batch) in enumerate(generator):
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        if (i + 1) * batch_size >= sample_count:
            break
    return features, labels


def main():
    data_base_dir = sys.argv[1]
    train_dir = os.path.join(data_base_dir, 'train')
    validation_dir = os.path.join(data_base_dir, 'validation')
    test_dir = os.path.join(data_base_dir, 'test')

    train_generator, validation_generator = \
        build_generators_conv_base(train_dir, validation_dir)

    model = build_conv_base_model()

    model.compile(optimizer=RMSprop(lr=2e-5),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    history = model.fit_generator(
        train_generator, steps_per_epoch=100, epochs=30,
        validation_data=validation_generator, validation_steps=50)
    model.save('models/cats_and_dogs_small_4.h5')
    metric_plots.plot_acc(history)


if __name__ == '__main__':
    assert len(sys.argv) == 2, \
        '1 argument required, {} given'.format(len(sys.argv) - 1)
    main()
