import json
import sys
import _thread
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def load_data(datapath): 
    data_frame = pd.read_csv(datapath, 
                            dtype={'tweet': 'object', 'emoji': 'object'}, 
                            encoding='utf-8', 
                            sep=',')
    data_frame.drop_duplicates().dropna(how='any')

    tweets = [str(tweet) for tweet in data_frame['tweet']]
    emojis = [emoji for emoji in data_frame['emoji']]

    data = {'tweet': tweets, 'emoji': emojis}
    data_frame = pd.DataFrame(data=data, dtype='object')

    return data_frame


def load_model(model_path, weights_path):
    with open(model_path, 'r') as model_file:
        architecture = json.load(model_file)
    model = model_from_json(architecture)
    model.load_weights(weights_path)

    return model


def prepare_input(text, tokenizer):
    X_pred = tokenizer.texts_to_sequences(text)
    X_pred = pad_sequences(X_pred, maxlen=50)
    X_pred = tf.convert_to_tensor(X_pred, dtype=tf.int64)

    return X_pred

def run(text, model, tokenizer, dataframe): 
    X_pred = prepare_input(text, tokenizer)
    prediction = model.predict(X_pred)
    emoji_index = np.argmax(prediction) 
    predicted_emoji = dataframe['emoji'][emoji_index]

    return predicted_emoji


def rerun(choice): 
    while choice == 'oui': 
        text = input('Votre texte : ')
        emoji = run(text, model, tokenizer, emoji_dataframe)
        print('The associated emoji is ', emoji)
        choice = input('Réeesayer ? ')
        rerun(choice)
    print('bye bye')
    sys.exit(0)

if __name__ == "__main__":
    emoji_dataframe = load_data('data/data.csv')
    model = load_model('models/bilstm_model.json', 'models/bilstm_weights.h5')
    tokenizer = Tokenizer(num_words=10000, 
                        filters='!"#$%&()*+,-./:;<=>?@[\\]^\'_`{|}~\t\n')
    tokenizer.fit_on_texts(emoji_dataframe['tweet'])    
    text = input('Votre texte : ')
    emoji = run(text, model, tokenizer, emoji_dataframe)
    print('The associated emoji is ', emoji)
    choice = input('Réeesayer ? ')
    rerun(choice)