import tensorflow as tf
import tensorflow_datasets as tfds

mnist_train, info = tfds.load("fashion_mnist", split="train", with_info="True")
assert isinstance(mnist_train, tf.data.Dataset)

print(info)

for item in mnist_train.take(1):
    print(type(item))
    print(item.keys())

    print(item['image'])
    print(item['label'])

    mnist_test, info = tfds.load("mnist", split="test", with_info="True")
    print(info)



