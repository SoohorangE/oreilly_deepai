import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import urllib.request

weights_url = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
weights_file = "inception_v3.h5"

urllib.request.urlretrieve(weights_url, weights_file)

pre_trained_model = InceptionV3(input_shape=(300, 300, 3),
                                include_top=False,
                                weights=None)

pre_trained_model.load_weights(weights_file)
pre_trained_model.summary()

for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7')
print('마지막 층의 출력 크기:', last_layer.output_shape)
last_output = last_layer.output

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(inputs=pre_trained_model.input, outputs=x)

model.compile(optimizer=RMSprop(learning_rate=1e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

training_dir = "horse-or-human/training/"
validation_dir = "horse-or-human/validation/"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_ds = train_datagen.flow_from_directory(
    training_dir,
    target_size=(300, 300),
    class_mode="binary",
    batch_size=20
)

for i in range(len(train_ds)):
    batch = train_ds[i][0]  # 이미지 데이터 가져오기
    if np.isnan(batch).any():
        print(f"🚨 NaN 값이 포함된 배치: {i}")
        break

print(train_ds.class_indices)

valid_datagen = ImageDataGenerator(rescale=1./255)
valid_ds = valid_datagen.flow_from_directory(
    validation_dir,
    target_size=(300, 300),
    class_mode="binary",
    batch_size=20
)

for i in range(len(valid_ds)):
    batch = valid_ds[i][0]  # 이미지 데이터 가져오기
    if np.isnan(batch).any():
        print(f"🚨 NaN 값이 포함된 배치: {i}")
        break

print(valid_ds.class_indices)

model.fit(train_ds, epochs=40, validation_data=valid_ds, verbose=1)

model.save('inception_v3_custom.keras')