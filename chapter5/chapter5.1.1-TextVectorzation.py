import tensorflow as tf
from tensorflow import keras

sentences = [
    'Today is a sunny day',
    'Today is a rainy day'
]

tv = keras.layers.TextVectorization(max_tokens=100)

tv.adapt(sentences)
print(tv.get_vocabulary())

test_data = [
    'Today is a snowy day',
    'Will it be rainy tomorrow?'
]

test_seq = tv(test_data)
test_seq.numpy()
print(test_seq)