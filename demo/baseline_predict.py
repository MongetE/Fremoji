import json
import numpy as np
import pandas as pd
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def load_data(datapath): 
    data_frame = pd.read_csv(datapath, 
                            dtype={'tweet': 'object', 'emoji': 'object'}, 
                            encoding='utf-8', 
                            sep=',')
    data_frame.drop_duplicates().dropna(how='any')

    return data_frame


def load_model(model_path, weights_path):
    with open(model_path, 'r') as model_file:
        architecture = json.load(model_file)
    model = model_from_json(architecture)
    model.load_weights(weights_path)

    return model


def prepare_input(text): 
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    X_pred = tokenizer.texts_to_sequences(text)
    X_pred = pad_sequences(X_pred, maxlen=50)

    return X_pred

def run(text, model, dataframe): 
    X_pred = prepare_input(text)
    prediction = model.predict(X_pred)

    emoji_index = np.argmax(prediction) 
    predicted_emoji = dataframe['emoji'][emoji_index]

    return predicted_emoji


if __name__ == "__main__":
    text = input('Votre texte : ')
    emoji_dataframe = load_data('data/processed.csv')
    model = load_model('models/baseline_model.json', 'models/baseline_weights.h5')
    emoji = run(text, model, emoji_dataframe)
