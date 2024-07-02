import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
import string
import re
import streamlit as st
from variables import chatwords

LABELS = ['negative', 'positive']
      
# Load the trained model
model = load_model('sentiment.h5')

# Load the saved Tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


def remove_html(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def remove_url(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)


def remove_punctuations(text):
    punctuations = string.punctuation
    no_punctuations = "".join(
        [char for char in text if char not in punctuations])
    return no_punctuations


def chat_conversation(text):
    new_text = []
    for word in text.split():
        if word.upper() in chatwords:
            new_text.append(chatwords[word.upper()])
        else:
            new_text.append(word)
    return " ".join(new_text)

# Preprocess user input


def preprocess(text):
    text = chat_conversation(text)
    text = text.lower()
    text = remove_html(text)
    text = remove_url(text)
    text = remove_punctuations(text)
    return text


def prepare_input(text):
    preprocessed_text = preprocess(text)
    sequences = tokenizer.texts_to_sequences([preprocessed_text])
    padded_sequences = pad_sequences(sequences, maxlen=200)
    return padded_sequences


# Define Streamlit app
st.set_page_config(page_title="Sentiment Analysis App",
                   page_icon=":smiley:", layout="wide")
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("Sentiment Analysis App")
st.sidebar.markdown(
    "Analyze the sentiment of your text with our state-of-the-art model.")
st.sidebar.markdown("## Instructions")
st.sidebar.markdown("""
1. Enter a sentence in the text area below.
2. Click the 'Analyze' button to get the sentiment.
3. See the result with an emotion icon and confidence score.
""")

st.title('Sentiment Analysis App')
st.markdown(
    "Analyze the sentiment of your text and see if it's positive or negative!")


def main():
    """
    Runs the Sentiment Analysis App.

    This function displays a Streamlit app that allows users to input a sentence and analyze its sentiment.
    The app prompts the user to enter a sentence using a text input field. When the user clicks the "Analyze" button,
    the function tokenizes the input text, passes it through a pre-trained model, and predicts the sentiment of the sentence.
    The predicted sentiment and the confidence score are then displayed on the app.

    Parameters:
    None

    Returns:
    None
    """

    # User input
    text_input = st.text_area(
        'Enter a sentence:', placeholder='Movie was fantastic!')

    # Analyze sentiment
    if st.button('Analyze'):
        if text_input:
            inputs = prepare_input(text_input)
            output = model.predict(inputs)
            # Get the prediction probability and class
            pred_prob = output[0][0]
            sentiment = LABELS[int(pred_prob > 0.5)]
            confidence = f"{pred_prob:.2f}" if pred_prob > 0.5 else f"{(1 - pred_prob):.2f}"

            # Display sentiment with emoticon
            if sentiment == 'positive':
                st.write(
                    f"### The review is: {sentiment} :smile: with {confidence} confidence.")
            else:
                st.write(
                    f"### The review is: {sentiment} :disappointed: with {confidence} confidence.")

            # Additional feedback
            if sentiment == 'positive':
                st.balloons()
                st.write("That's great to hear! Keep spreading positivity.")
            else:
                st.write("Sorry to hear that. Hope things get better!")


if __name__ == '__main__':
    main()
