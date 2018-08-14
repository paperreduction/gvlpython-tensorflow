# Word Vector Demo

# from gensim.models import KeyedVectors
#
# model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
#
# vector_for_easy = model['easy']
#
# print(vector_for_easy.shape)
# print(vector_for_easy)

# Tensorflow w/ Keras Demo
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dropout, Dense

MAX_NB_WORDS = 200000
MAX_SEQUENCE_LENGTH = 5000
EMBEDDING_DIM = 300

data_frame = pd.read_csv('demo_data.csv')

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
texts = [row['text'] for i, row in data_frame.iterrows()]
tokenizer.fit_on_texts(texts)

# Convert the words (strings) to sequence of word indices
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found {} unique tokens'.format(len(word_index)))

# Left pad everything to have the same length (5000 words)
sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# Our labels are binary. I eventually want to have multi-class with "to_categorical"
labels = np.asarray([row['family'] for i, row in data_frame.iterrows()])
print('Shape of data tensor:', sequences.shape)
print('Shape of label tensor:', labels.shape)

# Setup Word to Vec model
word2vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))

for word, i in word_index.items():
    try:
        embedding_vector = word2vec[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        # words not found in embedding index will be all-zeros.
        pass

print("The shape of the embedding matrix is {}".format(embedding_matrix.shape))

# Keras Neural Network

model = Sequential()
model.add(
    Embedding(
        input_dim=len(word_index)+1,    # Word index has min index 1
        output_dim=EMBEDDING_DIM,       # Output dimension will be 300 (vector)
        weights=[embedding_matrix],     # Our word vectors form the embedding weights
        input_length=MAX_SEQUENCE_LENGTH,
        mask_zero=True,                 # We're using 0 as a sign for missing.
        trainable=False,                # We don't want keras to adjust the embeddings
    )
)
model.add(LSTM(100))                        # LSTM layer after the embedding
model.add(Dropout(0.3))                     # Dropout layer helps prevent overfitting
model.add(Dense(1, activation='sigmoid'))   # Dense layer is fully connected feed forward

print(model.summary())

print('Compiling the model...')

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print('Done!')

# Do the training
model.fit(x_train, y_train, batch_size=5, epochs=10, validation_data=(x_test, y_test), shuffle=True)

# Does it work?
predictions = model.predict_classes(
    pad_sequences(
        tokenizer.texts_to_sequences(
            ["Quality time is a lie. Quantity time is what counts when you think about your time at home.",
             "We accept something as true when we have confidence.",
             "We’re going to be in Deuteronomy, chapter 6."]),
        maxlen=MAX_SEQUENCE_LENGTH))

# Output
print(predictions)
# array([[1], [0], [0]], dtype=int32)
