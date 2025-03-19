import tensorflow as tf

from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop

import numpy as np

training_dir = "horse-or-human/training/"
validation_dir = "horse-or-human/validation/"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_ds = train_datagen.flow_from_directory(
    training_dir,
    target_size=(300, 300),
    class_mode="binary"
)

print(train_ds.class_indices)

valid_datagen = ImageDataGenerator(rescale=1./255)

valid_ds = valid_datagen.flow_from_directory(
    validation_dir,
    target_size=(300, 300),
    class_mode="binary"
)

print(valid_ds.class_indices)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

model.summary()
model.fit(train_ds, epochs=15, validation_data=valid_ds)
model.save("horse-or-human.keras")

