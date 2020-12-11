import json
import nltk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Bidirectional, Dense, Embedding, Flatten, Input, LSTM, SpatialDropout1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

MAX_NB_WORDS = 50000
MAXLEN = 50
EMBEDDING_DIM = 250

if __name__ == "__main__":
    emoji_df = pd.read_csv("data/data.csv", sep=",", 
                        dtype={"tweet": "object", "emoji": "object"}, 
                        encoding="utf-8")
    emoji_df.drop_duplicates().dropna(how = 'any')

    tweets = [str(tweet) for tweet in emoji_df['tweet']]
    tokenizer = Tokenizer(num_words=10000, 
                        filters='!"#$%&()*+,-./:;<=>?@[\\]^\'_`{|}~\t\n')
    tokenizer.fit_on_texts(tweets)
    X = tokenizer.texts_to_sequences(tweets)
    X = pad_sequences(X, maxlen=MAXLEN)
    y = pd.get_dummies(emoji_df.emoji.values)
    X_train, X_test, y_train, y_test = train_test_split(X,y, 
                                                        test_size = 0.20, 
                                                        random_state = 42)

    model = Sequential()
    # Suppose to allow input of variable lengths afterwards
    model.add(Input(shape=[50], dtype=tf.int64, ragged=True))
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAXLEN))
    model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(LSTM(25, dropout=0.2, recurrent_dropout=0.2, 
                            return_sequences=True)))
    model.add(Bidirectional(LSTM(30, dropout=0.2, recurrent_dropout=0.2, 
                            return_sequences=True)))
    model.add(Bidirectional(LSTM(25, recurrent_dropout=0.2)))
    model.add(Dense(27, activation='relu'))
    model.add(Dense(25, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', 
                  metrics=['acc'])

        ## Uncomment below to train
    # history = model.fit(X_train, y_train, epochs=3, 
    #                 batch_size=64, validation_split=0.1)

    # acc = history.history['accuracy']
    # val_acc = history.history['val_accuracy']
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']

    # epochs = range(1, len(acc) + 1)

    # plt.plot(epochs, acc, 'bo', label='Training accuracy')
    # plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    # plt.title('Training and validation accuracy')
    # plt.legend()

    # plt.figure()

    # plt.plot(epochs, loss, 'bo', label='Training loss')
    # plt.plot(epochs, val_loss, 'b', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.legend()

    # plt.show()

        ## Uncomment below to train and save 
    # model.fit(X, y, epochs=3, batch_size=64)

    # saved_model = model.to_json()
    # model.save_weights('models/tmp_bilstm_weights.h5')
    # print('Weights saved to disk')

    # with open('models/tmp_bilstm_model.json', 'w') as model_file:
    #     json.dump(saved_model, model_file)
    # print('Model saved to disk')

    model.load_weights('models/tmp_bilstm_weights.h5')
    model.evaluate(X,y)