import os
import sys
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Input, Dropout, Flatten, Conv2D
from keras import models
from keras import optimizers
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras import regularizers
import gc
from keras import layers


def cnn_model():
    input_data = Input(shape=(150, 150, 3))
    x = Conv2D(32, (3, 3))(input_data)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dropout(0.2)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=input_data, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    return model


if __name__ == '__main__':
    base_dir = './cats_and_dogs_small'
    sys.path.append(base_dir)
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')
    train_cats_dir = os.path.join(train_dir, 'cats')

    # 数据扩增
    train_datagen = ImageDataGenerator(
        rescale=1 / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode="binary"
    )
    # 仅对train_data做数据扩增
    test_datagen = ImageDataGenerator(rescale=1 / 255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode="binary"
    )
    print("data is ready")
    gc.collect()

    model = cnn_model()
    for i in range(100):
        print("iterantion{}".format(i))
        history = model.fit_generator(
            train_generator,
            steps_per_epoch=100,
            epochs=1,
            validation_data=test_generator,
            validation_steps=50)
# val_loss: 0.5784 - val_acc: 0.8445
