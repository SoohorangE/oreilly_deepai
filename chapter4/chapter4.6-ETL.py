import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa

import math

#모델 정의 시작

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

model.compile(optimizer="Adam",
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 추출 단계

data = tfds.load("horses_or_humans", split="train", as_supervised=True)
val_data = tfds.load("horses_or_humans", split="test", as_supervised=True)

def augmentimages(image, label):

    image = tf.cast(image, tf.float32) / 255.0  # 정규화
    image = tf.image.random_flip_left_right(image)  # 좌우 반전
    image = tfa.image.rotate(image, math.radians(40), interpolation="NEAREST")  # 회전

    return image, label

train = data.map(augmentimages)
train_batches = train.shuffle(100).batch(32)
validation_batches = val_data.batch(32)

history = model.fit(train_batches, epochs=10, validation_data=validation_batches)
