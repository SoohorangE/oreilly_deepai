import os
import gdown
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow.keras.models import load_model

model = load_model('horse-or-human.keras')

base_url = "https://github.com/rickiepark/aiml4coders/raw/main/ch03/"

for i in range(1,4):
    file_name = "hh_image_{}.jpg".format(i)
    if not os.path.exists(file_name):
        print(f"다운로드 {file_name}...")
        gdown.download(base_url + file_name, file_name, quiet=False)
    else:
        print(f'{file_name} 이미 있어서 건너 뜁니다.')

sample_images = ['hh_image_{}.jpg'.format(i) for i in range(1,4)]

for fn in sample_images:

    plt.imshow(mpimg.imread(fn))
    plt.show()

    img = tf.keras.utils.load_img(fn, target_size=(300, 300))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255.0

    classes = model.predict(x)

    print(classes.shape)

    print("모델 출력:", classes[0][0])

    if classes[0][0] > 0.5:
        print(fn+"는 사람 입니다.")
    else:
        print(fn+"는 말 입니다.")
    print("-------------")
