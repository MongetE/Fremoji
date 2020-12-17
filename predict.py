import json
import click
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.text import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences


def load_utils(tokenizer_path, labels_path, index_path):
    with open(tokenizer_path, 'r', encoding='utf-8') as jsonfile:
        tokenizer_data = json.load(jsonfile)

    tokenizer = tokenizer_from_json(tokenizer_data)
    
    with open(labels_path, 'r') as jsonfile:
        reverse_labels = json.load(jsonfile)

    with open(index_path, 'r') as jsonfile:
        index = json.load(jsonfile)

    return tokenizer, reverse_labels, index


def prepare_input_data(text, tokenizer, maxlen):
    X_pred = tokenizer.texts_to_sequences(text)
    X_pred = pad_sequences(X_pred, maxlen=maxlen)

    return X_pred


def run_prediction(model, X, index, reverse_labels): 
    probalities = model.predict(X)
    prediction = np.argmax(probalities)
    for emoji, emoji_indices in index.items():
        if prediction in emoji_indices:
            predicted_emoji = reverse_labels[emoji]
    
    return predicted_emoji


@click.command()
@click.option('--tokenizer', help="Path to tokenizer json", 
              default='utils/baseline_tokenizer.json')
@click.option('--reverse_index', help="Path to the reverse index json", 
              default='utils/reverse_labels_baseline.json')
@click.option('--index', help='Path to the index json', 
              default='utils/baseline_emojis_indices.json')
@click.option('--model', help="Path to the model h5", 
              default='models/baseline.h5')
def run(tokenizer, reverse_index, index, model):
    text = input('Enter some text: ')
    tokenizer, reverse_labels, index = load_utils(tokenizer, reverse_index,
                                                  index)
    model = load_model(model)
    X_pred = prepare_input_data(text, tokenizer, 50)
    emoji = run_prediction(model=model, X=X_pred, index=index, reverse_labels=reverse_labels)
    print(f'{text} {emoji}')

if __name__ == "__main__":
    run()
    