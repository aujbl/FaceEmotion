import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


def cnn():
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
    return model


model = cnn()
model.load_weights('./first_try.h5')

test_data_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_data_gen.flow_from_directory(
        './test/',
        target_size=(48, 48),
        color_mode="rgb",
        shuffle=False,
        class_mode='categorical',
        batch_size=64)

filenames = test_generator.filenames
nb_samples = len(filenames)
predict = model.predict_generator(test_generator)

cls_name = np.array(['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised'])
submit_df = pd.DataFrame({'name': filenames, 'label': cls_name[predict.argmax(1)]})
submit_df.head()
submit_df = submit_df[submit_df['name'].apply(lambda x: 'test' in x)]
submit_df['name'] = submit_df['name'].apply(lambda x: x.split('\\')[-1])
submit_df = submit_df.sort_values(by='name')
submit_df.to_csv('submit.csv', index=None)


