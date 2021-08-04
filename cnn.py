import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

model = Sequential()
model.add(Conv2D(16, (5, 5), padding='same', input_shape=(48, 48, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (5, 5), padding='same', ))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (5, 5), padding='same', ))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation="softmax"))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])

model.summary()


batch_size = 64

train_data_gen = ImageDataGenerator(rescale=1./255,
                                    # rotation_range=10,
                                    # width_shift_range=0.2,
                                    # height_shift_range=0.2,
                                    # horizontal_flip=True,
                                    validation_split=0.2)

val_data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = train_data_gen.flow_from_directory('D:/faceemotion/train/',
                                                color_mode='rgb',
                                                batch_size=batch_size,
                                                class_mode='categorical',
                                                target_size=(48, 48),
                                                shuffle=True,
                                                subset='training')

val_data = val_data_gen.flow_from_directory('D:/faceemotion/train/',
                                            color_mode='rgb',
                                            batch_size=batch_size,
                                            class_mode='categorical',
                                            target_size=(48, 48),
                                            shuffle=False,
                                            subset='validation')

# train_data_gen = ImageDataGenerator(
#     rescale=1. / 255, validation_split=0.2)
#
# val_data_gen = ImageDataGenerator(
#     rescale=1. / 255, validation_split=0.2)

model.fit_generator(
        train_data,
        epochs=200,
        validation_data=val_data,
        validation_steps=200)
model.save_weights('first_try.h5')

