# Fremoji

## ***Disclaimer***

This project was a mentored project, led in collaboration with 3 other sutdents.
Out of respect for their work, only my share of the work is disclosed in this
repository. As a result, I do not make the corpus publicly available, since I
was not in charge of its gathering. **It also means that the `train.py` file
cannot be run without a csv file [tweet, emoji].**

## Context

Predicting emojis was a shared task of the SemEval 2018, for Spanish and English.
Emojis can provide useful insights i,to a user's feeling and their contribution
to sentiment analysis is studied by Felbo et al (2017) or Eisner et al. (2016),
to name of few. Results of the models produced during the SemEval 2018 shared
task achieved an accuracy oscillated between 30 and 40%. The two models in this
repo achieve an accuracy of 33% for the baseline (a simple neural network) and
34% for the biLSTM model.  
To the best of our knowledge, in January 2020, no attempt had been made for French.
Thus, the goal of our project was to try and provide a model to do so.

### Supported emojis

We were asked to include the 25 most frequent emojis in our corpus as part of
the classification task (see below)

![emojis_list](emojis.png)

## Getting started

Although the models can't be trained or evaluated without the corpus,
prediction can be made through command line thanks to `predict.py` or via a
streamlit app.

### Requirements

- python 3.8.5
- numpy==1.18.5
- streamlit==0.72.0
- click==7.1.2
- tensorflow==2.3.1
- pandas==1.1.4
- Keras==2.4.3
- matplotlib==3.3.3
- scikit_learn==0.23.2

### Running predictions

There are two ways to run the models in the streamlit to have fun predicting
emojis.

#### **With docker**

- Build the image running `docker build -t fremoji-image .`
- Run the container with `docker run --name fremoji -d -p 8501:8501 fremoji-image`
- Visit `localhost:8501`
- When you're done with the app, run `docker stop fremoji` to stop the container

#### **Inside your virtual environment**

- Run `pip install -r requirements.txt`
- Run `streamlit run app.py`
- A window should open at `localhost:8501` or visit `localhost:8501`  
or
- Run `predict.py`. By default, it runs using the baseline
model.  
To switch to the bilstm model, run `predict.py --model models/bilstm.h5
--reverse_index models_utils/reverse_labels_bilstm.json
--index models_utils/bilstm_emojis_indices.json`

## References

Eisner, B., Rocktäschel, T., Augenstein, I., Bošnjak, M., & Riedel, S. (2016).
emoji2vec: Learning emoji representations from their description.
[arXiv preprint arXiv:1609.08359](https://arxiv.org/abs/1609.08359).

Felbo, B., Mislove, A., Søgaard, A., Rahwan, I., & Lehmann, S. (2017). Using
millions of emoji occurrences to learn any-domain representations for detecting
sentiment, emotion and sarcasm.
[arXiv preprint arXiv:1708.00524](https://arxiv.org/abs/1708.00524).
