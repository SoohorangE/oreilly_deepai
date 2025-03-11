import sys
import gdown

import joblib

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing import image

model = joblib.load('./horse-training.pkl')

base_url = "https://github.com/rickiepark/aiml4coders/raw/main/ch03/"

for i in range(1,4):
    gdown.download(base_url + "hh_image_{}.jpg".format(i))

sample_images = ['hh_image_{}.jpg'.format(i) for i in range(1,4)]

for fn in sample_images:

    plt.imshow(mpimg.imread(fn))
    plt.show()

    img = tf.keras.utils.load_img(fn, target_size=(300, 300))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    classes = model.predict(x)

    print("모델 출력:", classes[0][0])

    if classes[0][0] > 0.5:
        print(fn+"는 사람 입니다.")
    else:
        print(fn+"는 말 입니다.")
    print("-------------")
