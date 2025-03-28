import os
import gdown

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras.preprocessing import image
from tensorflow.keras.models import load_model

model = load_model('rsp.keras')

base_url = "https://github.com/rickiepark/aiml4coders/raw/main/ch03/"

for i in range(1, 4):
    file_name = "rsp_image_{}.jpg".format(i)
    if not os.path.exists(file_name):  # 파일이 없으면 다운로드
        print(f"다운로드 {file_name}...")
        gdown.download(base_url + file_name, file_name, quiet=False)
    else:
        print(f"{file_name} 이미 있어서 넘어갑니다.")

sample_images = ['rsp_image_{}.jpg'.format(i) for i in range(1, 4)]

rsp_name = ['paper', 'rock', 'scissors']

for fn in sample_images:

    plt.imshow(mpimg.imread(fn))
    plt.show()

    img = tf.keras.utils.load_img(fn, target_size=(150, 150))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255.0

    classes = model.predict(x)

    print(classes)

    idx = np.argmax(classes[0])
    print(fn + "는 {} 입니다.".format(rsp_name[idx]))

    print("-------------")
