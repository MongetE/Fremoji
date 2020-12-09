import streamlit as st 
from demo import baseline_predict as bsp

if __name__ == "__main__":
    emoji_dataframe = bsp.load_data('data/processed.csv')

    model_to_used = st.selectbox('Select a model', ('Baseline', ))
    text_input = st.text_input('Your tweet: ')

    if model_to_used == 'Baseline': 
        model = bsp.load_model('models/baseline_model.json', 'models/baseline_weights.h5')

    if len(text_input) > 1:
        emoji = bsp.run(text_input, model, emoji_dataframe)
        emoji_string = f"The associated emoji is {emoji}."
        st.write(emoji_string)
    else: 
        st.write('Enter some text, hit enter and wait for the surprise ! 😂')