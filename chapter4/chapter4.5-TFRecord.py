import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core.features import Image, ClassLabel, FeaturesDict

data, info = tfds.load("mnist", with_info=True)
print(info)

features = FeaturesDict({
    'image': Image(shape=(28, 28, 1), dtype=tf.uint8),
    'label': ClassLabel(num_classes=10),
})
