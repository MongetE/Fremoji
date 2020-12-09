import json
import time
import nltk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


def load_data(datapath): 
    data_frame = pd.read_csv(datapath, 
                            dtype={'tweet': 'object', 'emoji': 'object'}, 
                            encoding='utf-8', 
                            sep=',')
    data_frame.drop_duplicates().dropna(how='any')

    return data_frame


def prepare_input_output_data(tokenizer, dataframe, max_words, len_sequence): 
    X = tokenizer.texts_to_sequences(dataframe['tweet'])
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    y = pd.get_dummies(dataframe['emoji'])

    print('Shape of data tensor:', X.shape)
    print('Shape of label tensor:', y.shape)    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42)

    return X, y, X_train, X_test, y_train, y_test


def get_word_embedding(embedding_path, tokens, max_words, embedding_dimension):
    embedding_index = {}

    with open(embedding_path, 'r', encoding='utf-8') as embedding_file: 
        for line in embedding_file: 
            values = line.split()
            word = values[0]
            vectors = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = vectors

    # Can be replaced with Tokenizer.word_index
    tokens = list(set(tokens))
    tokens_index = {tokens[i]: i for i in range(len(tokens))}

    embedding_matrix = np.zeros((max_words, embedding_dimension))
    for token, i in tokens_index.items(): 
        try: 
            embedding_vector = embedding_index.get(token)
            if i < max_words:
                if embedding_vector is not None: 
                    embedding_matrix[i] = embedding_vector
        except KeyError: 
            continue

    return embedding_matrix


def build_model(max_words, embedding_dimension, max_sequence, 
                embedding_matrix=None):
    # Define model architecture 
    model = Sequential()
    model.add(Embedding(max_words, embedding_dimension, input_length=max_sequence))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(25, activation='sigmoid'))

    # Add word embedding to Embedding layers
    # They are not updated to avoid messing with what is already learnt
    # model.layers[0].set_weights([embedding_matrix])
    # model.layers[0].trainable = False

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', 
                metrics=['acc'])

    return model

if __name__ == "__main__":
    tweet_dataframe = load_data('data/processed.csv')
    embedding_path = 'embeddings/ft_cc.fr.300.vec'

    # Can be replaced with Tokenizer.word_index to get nb unique tokens
    tweets_concatenated = ""
    for row in tweet_dataframe['tweet']:
        tweets_concatenated += row + ' '
    tokens = nltk.word_tokenize(tweets_concatenated)

    MAX_NB_WORDS = round(len(set(tokens))/2)
    # Tweet can't be much longer than 50 tokens
    MAX_SEQUENCE_LENGTH = 50
    EMBEDDING_DIMENSION = 300

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(tweets_concatenated)

    X, y, X_train, X_test, y_train, y_test = prepare_input_output_data(tokenizer, 
                                                                tweet_dataframe, 
                                                                MAX_NB_WORDS, 
                                                                MAX_SEQUENCE_LENGTH)
    print("Data is ready")

    # Pre-trained word embedding only slow the model and do not bring 
    # significative gain 

    # print("Building embedding matrix, this might take a while")
    # start = time.time()
    # embedding_matrix = get_word_embedding(embedding_path, tokens, 
    #                                       MAX_NB_WORDS, EMBEDDING_DIMENSION)
    # end = time.time() - start
    # print(f"Embedding matrix is built (task completed in {end:.2f}s)")

    model = build_model(MAX_NB_WORDS, EMBEDDING_DIMENSION, 
                        MAX_SEQUENCE_LENGTH)
    print("Model is ready to be trained")

    # Uncomment to evaluate
    # history = model.fit(X_train, y_train, epochs=5, batch_size=64, 
    #           validation_data=(X_test, y_test))
    

    # acc = history.history['acc']
    # val_acc = history.history['val_acc']
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']

    # epochs = range(1, len(acc) + 1)

    # plt.plot(epochs, acc, 'bo', label='Training acc')
    # plt.plot(epochs, val_acc, 'b', label='Validation acc')
    # plt.title('Training and validation accuracy')
    # plt.legend()

    # plt.figure()

    # plt.plot(epochs, loss, 'bo', label='Training loss')
    # plt.plot(epochs, val_loss, 'b', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.legend()

    # plt.show()

    model.fit(X, y, epochs=3, batch_size=64)

    saved_model = model.to_json()
    model.save_weights('models/baseline_weights.h5')
    print('Weights saved to disk')

    with open('models/baseline_model.json', 'w') as model_file:
        json.dump(saved_model, model_file)
    print('Model saved to disk')