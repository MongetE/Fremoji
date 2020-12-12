import json
import pickle
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def load_data(datapath): 
    with open(datapath, 'r', encoding='utf-8') as jsonfile:
        emojis = json.load(jsonfile)
    
    return emojis

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

def run(text, model, tokenizer, emojis): 
    X_pred = prepare_input(text, tokenizer)
    prediction = model.predict(X_pred)
    emoji_index = np.argmax(prediction) 
    for emoji, emoji_indices in emojis.items():
        if emoji_index in emoji_indices:
            predicted_emoji = emoji
    
    return predicted_emoji


def rerun(choice): 
    while choice == 'oui': 
        text = input('Votre texte : ')
        emoji = run(text, model, tokenizer, emojis)
        print('The associated emoji is ', emoji)
        choice = input('Réeesayer ? (oui|non)')
        rerun(choice)
    print('bye bye')
    sys.exit(0)

if __name__ == "__main__":
    emojis = load_data('data/emojis.json')

    with open('models/tokenizer.pickle', 'rb') as picklefile:
        tokenizer = pickle.load(picklefile)
    model = load_model('models/baseline_model.json', 'models/baseline_weights.h5')
   
    text = input('Votre texte : ')
    tokenizer.fit_on_texts(text)    
    emoji = run(text, model, tokenizer, emojis)
    print('The associated emoji is ', emoji)
    choice = input('Réeesayer ? (oui|non) ')
    rerun(choice)