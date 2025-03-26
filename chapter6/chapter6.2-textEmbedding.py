import json
import tensorflow as tf
from keras import Sequential

from keras.preprocessing.text import Tokenizer
from bs4 import BeautifulSoup
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense

import numpy as np

from tb import TensorBoard, make_TensorBoard

dir_name = "log_deepai"
TensorB = make_TensorBoard(dir_name)

sentences = []
labels = []
urls = []

# 불용어
stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
             "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",
             "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
             "he", "hed", "hes", "her", "here", "heres", "hers", "herself", "him", "himself", "his", "how",
             "hows", "i", "id", "ill", "im", "ive", "if", "in", "into", "is", "it", "its", "itself",
             "lets", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought",
             "our", "ours", "ourselves", "out", "over", "own", "same", "she", "shed", "shell", "shes", "should",
             "so", "some", "such", "than", "that", "thats", "the", "their", "theirs", "them", "themselves", "then",
             "there", "theres", "these", "they", "theyd", "theyll", "theyre", "theyve", "this", "those", "through",
             "to", "too", "under", "until", "up", "very", "was", "we", "wed", "well", "were", "weve", "were",
             "what", "whats", "when", "whens", "where", "wheres", "which", "while", "who", "whos", "whom", "why",
             "whys", "with", "would", "you", "youd", "youll", "youre", "youve", "your", "yours", "yourself",
             "yourselves"]

with open('Sarcasm_Headlines_Dataset.json', 'r') as f:
    datastore = json.load(f)
    for item in datastore:
        sentence = item['headline'].lower()

        sentence = sentence.replace(",", " , ")
        sentence = sentence.replace(".", " . ")
        sentence = sentence.replace("-", " - ")
        sentence = sentence.replace("/", " / ")

        soup = BeautifulSoup(sentence, "html.parser")
        sentence = soup.get_text()
        words = sentence.split()

        filtered_sentence = ""
        for word in words:
            if word not in stopwords:
                filtered_sentence += word + " "

        sentences.append(filtered_sentence)
        labels.append(item['is_sarcastic'])
        urls.append(item['article_link'])

training_size = 23000

training_sentences = sentences[0:training_size]
training_labels = labels[0:training_size]
test_sentences = sentences[training_size:]
test_labels = labels[training_size:]

vocab_size = 20000
max_length = 10
trunc_type = "post"
padding_type = "post"
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length,
                                padding=padding_type, truncating=trunc_type)

test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, maxlen=max_length,
                            padding=padding_type, truncating=trunc_type)

training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
test_padded = np.array(test_padded)
test_labels = np.array(test_labels)

model = Sequential([
    Embedding(10000, 16),
    GlobalAveragePooling1D(),
    Dense(24, activation="relu"),
    Dense(1, activation="sigmoid"),
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

model.fit(training_padded, training_labels, epochs=30, validation_data=(test_padded, test_labels), callbacks=[TensorB])




