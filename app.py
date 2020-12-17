import numpy as np
import streamlit as st
import tensorflow.python.keras.backend as be
from keras.models import load_model
from predict import load_utils, prepare_input_data, run_prediction


@st.cache(allow_output_mutation=True)
def load(model_path, tokenizer_path, labels_path, index_path):
    tokenizer, reverse_labels, index = load_utils(tokenizer_path, 
                                                  labels_path, 
                                                  index_path)
    model = load_model(model_path)
    session = be.get_session()

    return tokenizer, reverse_labels, index, model, session


@st.cache()
def load_examples():
    with open('data/streamlit_examples.txt', 'r', encoding='utf-8') as txt_file:
            examples = tuple(txt_file.readlines())
    
    return examples


def sidebar():
    st.sidebar.markdown('# About')
    st.sidebar.markdown('Fremoji was a mentored project aiming to predict \
                        emojis in French tweets')
    st.sidebar.markdown('_Note_: The examples are real tweets, taken either \
                        from our corpus or in the context of Covid-19 in \
                        December 2020.')
    st.sidebar.markdown('## Models')
    st.sidebar.markdown('Two models can be used to predict emojis. The BiLSTM\
                        model is supposed to perform slightly better than the\
                        baseline. However, people use emojis very differently\
                        : Barbieri et al. (2017) state that the way emojis are\
                        used, or even the most used emojis, varies from one\
                        language to another, and variations also occurs within\
                        a linguistic area. As such, any predicted emoji for a\
                        given tweet could actually be used by someone, making\
                        the evaluation process rather dubious.')
    st.sidebar.markdown('# References')
    st.sidebar.markdown('Barbieri, F., Ballesteros, M., & Saggion, H. (2017). \
                        Are emojis predictable?. arXiv preprint arXiv:1702.07285.')
    st.sidebar.markdown('# See also')
    st.sidebar.markdown('[Fremoji github repo]')


def show_prediction(model, tokenizer, reverse_labels, index, text, session):
    X_pred = prepare_input_data(text, tokenizer, 50)
    be.set_session(session)
    emoji = run_prediction(model=model, X=X_pred, index=index, 
                           reverse_labels=reverse_labels)
    st.markdown('## Your tweet')
    tweet_string = f'{text} {emoji}'
    st.write(tweet_string)


def body():
    st.markdown('# Fremoji 😏')
    st.write('Predict emojis in French tweets! ')
    st.markdown('_It takes a few seconds to output an emoji. Further\
                information about the project in the sidebar !_')
    model_to_used = st.selectbox('Select a model', ('Baseline', 'BiLSTM'))
    st.write('Enter some text, hit enter and wait for the surprise ! 😮')
    text_input = st.text_input('Your tweet: ')
    st.write('or')

    example = st.selectbox('Choose example (clear any text in the box above)', examples)

    if model_to_used == 'Baseline': 
        model = bs_model
        tokenizer = bs_tokenizer
        reverse_labels = bs_reverse_labels
        index = bs_index
        session = bs_session
    elif model_to_used == 'BiLSTM':
        model = bi_model
        tokenizer = bi_tokenizer
        reverse_labels = bi_reverse_labels
        index = bi_index
        session = bi_session

    text = ""
    if len(text_input) > 1:
        text = text_input
    elif len(example) > 1:
        text = example

    if len(text) > 0:
        show_prediction(model=model, tokenizer=tokenizer, text=text, 
                        reverse_labels=reverse_labels, index=index, session=session)

if __name__ == "__main__":
    bi_tokenizer, bi_reverse_labels, bi_index, bi_model, bi_session = load('models/bilstm.h5', 
                                                                          'models_utils/bilstm_tokenizer.json', 
                                                                          'models_utils/reverse_labels_bilstm.json', 
                                                                          'models_utils/bilstm_emojis_indices.json') 
    bs_tokenizer, bs_reverse_labels, bs_index, bs_model, bs_session = load('models/baseline.h5', 
                                                                          'models_utils/baseline_tokenizer.json', 
                                                                          'models_utils/reverse_labels_baseline.json', 
                                                                          'models_utils/baseline_emojis_indices.json')                                                                       
    examples = load_examples()
    body()
    sidebar()