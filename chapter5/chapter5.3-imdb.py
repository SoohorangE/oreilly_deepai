import tensorflow as tf
import tensorflow_datasets as tfds

from keras.preprocessing.text import Tokenizer
from bs4 import BeautifulSoup
import string

imdb_sentences = []
train_data = tfds.as_numpy(tfds.load("imdb_reviews", shuffle_files=True, split="train"))
for item in train_data:
    imdb_sentences.append(str(item['text']))

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(imdb_sentences)
sequences = tokenizer.texts_to_sequences(imdb_sentences)

# 코드 완성 X
