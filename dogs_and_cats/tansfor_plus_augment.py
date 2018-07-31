import os
import sys

from keras import layers
from keras import models
from keras import optimizers
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator

if __name__ == '__main__':
    base_dir = './cats_and_dogs_small'
    sys.path.append(base_dir)
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')
    train_cats_dir = os.path.join(train_dir, 'cats')

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,  # 代替rescale
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(150, 150, 3))
    conv_base.trainable = True

    set_trainable = False
    for layer in conv_base.layers:
        if layer.name == 'block5_conv1':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(Dropout(0.2))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(layers.Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    print('This is the number of trainable weights '
          'after freezing the conv base:', len(model.trainable_weights))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=2e-5),
                  metrics=['acc'])

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=40,
        validation_data=validation_generator,
        validation_steps=50)
