import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'Today is a sunny day',
    'Today is a rainy day',
    'Is it sunny today?',
    'I really enjoyed walking in the snow today'
]

tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index

sentences = tokenizer.texts_to_sequences(sentences)
print(sentences)

# 기본 앞 부분에 패딩을 넣으려면
padded = pad_sequences(sentences)
# 뒷 부분에 패딩을 넣으려면
padded = pad_sequences(sentences, padding="post")
# 패딩(채우기)의 길이를 조절하려면
padded = pad_sequences(sentences, padding="post", maxlen=6)
# 위에 소스는 앞, 뒤가 잘리는데 문장의 끝 부분을 잘라내려면
padded = pad_sequences(sentences, padding="post", maxlen=6, truncating="post")
print(padded)
