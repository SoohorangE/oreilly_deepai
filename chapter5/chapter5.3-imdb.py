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

table = str.maketrans('', '', string.punctuation)

imdb_sentences = []

train_data = tfds.as_numpy(tfds.load("imdb_reviews", shuffle_files=True, split="train"))
for item in train_data:
    # 소문자로 변형
    sentence = str(item['text'].decode('utf-8').lower())

    sentence = sentence.replace(",", " , ")
    sentence = sentence.replace(".", " . ")
    sentence = sentence.replace("-", " - ")
    sentence = sentence.replace("/", " / ")

    soup = BeautifulSoup(sentence)
    sentence = soup.get_text()
    words = sentence.split()

    filtered_sentence = ""
    for word in words:
        word = word.translate(table)
        if word not in stopwords:
            filtered_sentence = filtered_sentence + word + " "
    imdb_sentences.append(filtered_sentence)

tokenizer = Tokenizer(num_words=90000)
tokenizer.fit_on_texts(imdb_sentences)
sequences = tokenizer.texts_to_sequences(imdb_sentences)
print(tokenizer.word_index)

# 테스트 문장
sentences = [
    'Today is a sunny day',
    'Today is a rainy day',
    'Is it sunny today?'
]

sequences2 = tokenizer.texts_to_sequences(sentences)
print(sequences2)

reverse_word_index = dict(
    [(value, key) for (key, value) in tokenizer.word_index.items()]
)

decoded_review = ' '.join([reverse_word_index.get(i, '?') for i in sequences2[0]])
print(decoded_review)



