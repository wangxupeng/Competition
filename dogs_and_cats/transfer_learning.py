import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
import sys
from keras import models
from keras import layers
from keras import optimizers
from tqdm import tqdm


def extract_features(directory, sample_count, conv_base):
    batch_size = 20
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in tqdm(generator):
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            # 当generators产生出循环中没有的数据时就要break了
            break
    return features, labels


if __name__ == '__main__':
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(150, 150, 3))

    base_dir = r'./cats_and_dogs_small'
    sys.path.append(base_dir)

    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')

    datagen = ImageDataGenerator(rescale=1. / 255)

    train_features, train_labels = extract_features(train_dir, 2000, conv_base)
    validation_features, validation_labels = extract_features(validation_dir, 1000, conv_base)
    test_features, test_labels = extract_features(test_dir, 1000, conv_base)

    train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
    validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
    test_features = np.reshape(test_features, (1000, 4 * 4 * 512))

    model = models.Sequential()
    model.add(layers.Dense(1024, activation='relu', input_dim=4 * 4 * 512))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    history = model.fit(train_features, train_labels,
                        epochs=30,
                        batch_size=20,
                        validation_data=(validation_features, validation_labels))
