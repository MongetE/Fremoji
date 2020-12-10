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


MAX_NB_WORDS = 50000
MAXLEN = 25
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
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAXLEN))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(25, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', 
                    metrics=['acc'])
    
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

    model.fit(X, y, epochs=2, batch_size=64)

    saved_model = model.to_json()
    model.save_weights('models/baseline_weights.h5')
    print('Weights saved to disk')

    with open('models/baseline_model.json', 'w') as model_file:
        json.dump(saved_model, model_file)
    print('Model saved to disk')

    