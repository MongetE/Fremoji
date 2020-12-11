import streamlit as st 
from keras.preprocessing.text import Tokenizer
from demo import baseline_predict as bsp

def init_constant_var():
    baseline = bsp.load_model('models/baseline_model.json', 
                              'models/baseline_weights.h5')
    bilstm = bsp.load_model('models/bilstm_model.json', 
                            'models/bilstm_weights.h5')
    emoji_dataframe = bsp.load_data('data/data.csv')
    tokenizer = Tokenizer(num_words=10000, 
                        filters='!"#$%&()*+,-./:;<=>?@[\\]^\'_`{|}~\t\n')
    tokenizer.fit_on_texts(emoji_dataframe['tweet'])

    with open('demo/streamlit_examples.txt', 'r', encoding='utf-8') as txt_file:
        examples = tuple(txt_file.readlines())

    return baseline, bilstm, emoji_dataframe, tokenizer, examples


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

    

def body():
    st.markdown('# Fremoji 😏')
    st.write('Predict emojis in French tweets! ')
    st.markdown('_It takes a few seconds to output an emoji. Further\
                information in the sidebar !_')
    model_to_used = st.selectbox('Select a model', ('Baseline', 'BiLSTM'))
    st.write('Enter some text, hit enter and wait for the surprise ! 😮')
    text_input = st.text_input('Your tweet: ')
    st.write('or')

    example = st.selectbox('Choose example (clear any text in the box above)', examples)

    if model_to_used == 'Baseline': 
        model = baseline
    elif model_to_used == 'BiLSTM':
        model = bilstm

    if len(text_input) > 1:
        emoji = bsp.run(text_input, model, tokenizer, emoji_dataframe)
        emoji_string = f"{text_input} {emoji}."
        st.markdown('## Your tweet')
        st.write(emoji_string)
    elif len(example) > 1:
        emoji = bsp.run(text=example, model=model, tokenizer=tokenizer, dataframe=emoji_dataframe)
        emoji_string = f"{example} {emoji}."
        st.markdown('## Your tweet')
        st.write(emoji_string)
        

if __name__ == "__main__":
    baseline, bilstm, emoji_dataframe, tokenizer, examples = init_constant_var()
    sidebar()
    body()
    