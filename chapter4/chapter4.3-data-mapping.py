import math
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa

# 데이터셋 로드
data = tfds.load("horses_or_humans", split="train", shuffle_files=True)

# 이미지 증강 함수
def augmentimages(sample):
    image = sample['image']
    label = sample['label']

    image = tf.cast(image, tf.float32) / 255.0  # 정규화
    image = tf.image.random_flip_left_right(image)  # 좌우 반전
    image = tfa.image.rotate(image, math.radians(40), interpolation="NEAREST")  # 회전

    return image, label


# 데이터 변환 적용
train = data.map(augmentimages)
train_batches = train.shuffle(100).batch(64)