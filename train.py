import json
import warnings
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from random import sample
from keras.models import Sequential
from keras.layers import Bidirectional, Dense, Embedding, Flatten,\
                         Input, LSTM, SpatialDropout1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')


def load_training_data(datapath):
    dataframe = pd.read_csv(datapath, sep=',', encoding='utf-8', 
                            dtype={'tweet': 'object', 
                                   'emoji': 'object'})
    dataframe.drop_duplicates().dropna(how='any')       

    return dataframe


def correct_datatype(dataframe):
    """
        Some tweets are considered as float by Tf so they are converted to str

        Parameters
        ----------
            dataframe: pandas.DataFrame
        
        Returns
        -------
            new_dataframe: pandas.DataFrame
                Same dataframe but with all row converted to string
            reverse_labels: dict
                Index encoding of the labels
    """

    tweets = [str(tweet) for tweet in dataframe['tweet']]
    emojis = [emoji for emoji in dataframe['emoji']]
    set_emojis = list(set(emojis))
    labels = {set_emojis[i]: i for i in range(len(set_emojis))}  
    reverse_labels = {index: emoji for emoji, index in labels.items()}
    emojis_as_labels = [labels[emoji] for emoji in emojis]
    data = {'tweets': [], 'emojis': []}
    for i in range(1, len(tweets)):
        data['tweets'].append(tweets[i])
        data['emojis'].append(emojis_as_labels[i])
    # data = {tweets[i]: emojis[i] for i in range(len(tweets))}    
    new_dataframe = pd.DataFrame(data=data) 

    return new_dataframe, reverse_labels 


def prepare_input_data(dataframe, save, maxlen=50, max_words=10000, model_name=None):
    """
        Prepapre the data to be fed to the model.

        Parameters
        ----------
            dataframe: pandas.DataFrame
            save: bool
            maxlen: int
                Maximum number of tokens per tweet
            max_words: int
                Maximum of words for the tokenizer
            model_name: str
                If save, save the variables needed for prediction, 
                including the name of the model in the output file
        
        Returns
        -------
            X: numpy.array
                Array of tweets fit for model input
            y: pandas.DataFrame
                Onehot encoding of the labels
    """
    tokenizer = Tokenizer(num_words=max_words, 
                        filters='!"#$%&()*+,-./:;<=>?@[\\]^\'_`{|}~\t\n')
    tokenizer.fit_on_texts(dataframe['tweets'])
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)  
    X = tokenizer.texts_to_sequences(dataframe['tweets'])
    X = pad_sequences(X, maxlen=maxlen)
    y = pd.get_dummies(dataframe['emojis'])

    if save: 
        saved_tokenizer = tokenizer.to_json()
        with open(f'models_utils/{model_name}_tokenizer.json', 'w', encoding='utf-8') as jsonfile:
            json.dump(saved_tokenizer, jsonfile, ensure_ascii=False)
        
        emojis = [emoji for emoji in dataframe['emojis']]
        emojis_indices = {}
        for i in range(len(emojis)):
            if emojis[i] in emojis_indices.keys():
                emojis_indices[emojis[i]].append(i)
            else:
                emojis_indices[emojis[i]] = [i]
        
        with open(f'models_utils/{model_name}_emojis_indices.json', 'w') as jsonfile:
            json.dump(emojis_indices, jsonfile)

    return X, y


def balance_weights(onehot_labels): 
    """
        Adjust weights so that they are inversely proportional to class 
        frequencies.

        Parameters
        ----------
            onehot_labels: pandas.DataFrame
                A one-hoy encoded dataframe such as output by 
                pandas.get_dummies()
        
        Returns
        -------
            class_weights: dict
                A dictionary of weights
    """
    # Classes are severly imbalanced so classes are given more or less weight
    # Although it actually lowers the model's ability to generalize
    classes_dataframe = onehot_labels.stack().reset_index()
    classes_occurrences = classes_dataframe[classes_dataframe[0] == 1]['level_1']
    class_weights = compute_class_weight('balanced', np.unique(classes_occurrences), 
                                         classes_occurrences)
    class_weights = dict(enumerate(class_weights))
    
    return class_weights


def build_baseline_model(max_words=10000, maxlen=50, embedding_dimension=250):
    model = Sequential()
    model.add(Input(shape=[50], dtype=tf.int64, ragged=True))
    model.add(Embedding(max_words, embedding_dimension, input_length=maxlen))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(25, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', 
                    metrics=['accuracy'])
    
    return model


def build_bilstm_model(max_words, maxlen, embedding_dimension):
    model = Sequential()
    # Suppose to allow input of variable lengths afterwards
    model.add(Input(shape=[50], dtype=tf.int64, ragged=True))
    model.add(Embedding(max_words, embedding_dimension, input_length=maxlen))
    model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(LSTM(25, dropout=0.2, recurrent_dropout=0.2, 
                            return_sequences=True)))
    model.add(Bidirectional(LSTM(30, dropout=0.2, recurrent_dropout=0.2, 
                            return_sequences=True)))
    model.add(Bidirectional(LSTM(25, recurrent_dropout=0.2)))
    model.add(Dense(27, activation='relu'))
    model.add(Dense(25, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', 
                  metrics=['accuracy'])

    return model


def build_training_plot(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


def train(dataframe, model_name, maxlen, max_words, embedding_dim, epochs, 
          batch_size, kfold, save, class_weights=None):
    accuracies = []
    if not save:
        for i in range(0, kfold): 
            if model_name == 'baseline':
                model = build_baseline_model(maxlen=maxlen, max_words=max_words, 
                                            embedding_dimension=embedding_dim)
            else:
                model = build_bilstm_model(maxlen=maxlen, max_words=max_words, 
                                        embedding_dimension=embedding_dim)

            X,y = prepare_input_data(dataframe=dataframe, maxlen=maxlen, save=save, 
                                     model_name=model)    

            if i == 0:
                model.summary()

            print(f'Training for fold {i+1}')
            if class_weights is not None:
                history = model.fit(X, y, validation_split=0.2, epochs=epochs, 
                                    batch_size=batch_size, class_weight=class_weights)
            else: 
                history = model.fit(X, y, validation_split=0.2, epochs=epochs, 
                                    batch_size=batch_size)
            
            accuracies.append(history.history['val_accuracy'])

        accuracies = np.asarray(accuracies)
        print(f'Training accuracy: {accuracies.mean()*100:.2f}% (+/- {accuracies.std()*2:.3f})')
        build_training_plot(history)

        model.evaluate(X,y)

    else: 
        if model_name == 'baseline':
                model = build_baseline_model(maxlen=maxlen, max_words=max_words, 
                                            embedding_dimension=embedding_dim)
        else:
            model = build_bilstm_model(maxlen=maxlen, max_words=max_words, 
                                    embedding_dimension=embedding_dim)
        model.summary()
        model.fit(X, y, epochs=epochs, batch_size=batch_size)
        model.save(f'models/{model_name}.h5')
        model.evaluate(X,y)


@click.command()
@click.option('--data_path', help='Path to the data file')
@click.option('--model', help='Name of the model to train', 
              type=click.Choice(['baseline', 'bilstm'], case_sensitive=False),
              default='baseline')
@click.option('--maxlen', help='Max number of tokens in text sequences', 
              default=50, type=int)
@click.option('--embedding_dim', help='Dimension for embedding layer', 
              default=250, type=int)
@click.option('--max_words', help="Maximum number of words for the Tokenizer\
              dictionary", default=10000, type=int)
@click.option('--epochs', help='Number of epochs', default=2, type=int)
@click.option('--batch_size', help='Number of training examples for a training\
              iteration', default=64, type=int)
@click.option('--balanced', help='If data is unbalanced, whether to adjust \
              weights inversely proportional to class frequencies (less frequent\
              class will be given more weight)', default=False, type=bool)
@click.option('--kfold', help='Number of fold for cross-validation', 
              default=2, type=int)
@click.option('--save', help='Whether to save the model once trained', 
              default=False, type=bool)
def run(data_path, model, maxlen, embedding_dim, max_words, epochs, batch_size, 
        balanced, kfold, save):
    dataframe = load_training_data(data_path)
    dataframe, reverse_labels = correct_datatype(dataframe)
    if save: 
        with open(f'models_utils/reverse_labels_{model}.json', 'w', encoding='utf-8') as jsonfile:
            json.dump(reverse_labels, jsonfile, ensure_ascii=False)

    if balanced:
        class_weights = balance_weights(y)
        train(dataframe=dataframe, model_name=model, maxlen=maxlen, kfold=kfold,
              embedding_dim=embedding_dim, epochs=epochs, max_words=max_words, 
              class_weights=class_weights, batch_size=batch_size, save=save)
    else:
        train(dataframe=dataframe, model_name=model, maxlen=maxlen, kfold=kfold,
              embedding_dim=embedding_dim, epochs=epochs, max_words=max_words, 
              batch_size=batch_size, save=save)


if __name__ == "__main__":
    run()
