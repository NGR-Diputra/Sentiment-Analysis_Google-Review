from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import string
import re
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sentiment_analysis_dukcapil import case_folding, load_abbreviation_file, normalize_text, stopwords_removal, stemming_text

app = Flask(__name__)

# Load the model and tokenizer
model = load_model('sentiment_model.h5')
with open('tokenizer_config.json') as config_file:
    config = json.load(config_file)
    max_words = config['num_words']
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.word_index = json.loads(config['word_index'])
    tokenizer.index_word = {str(i): word for word, i in tokenizer.word_index.items()}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        result = perform_sentiment_analysis(text)
        return render_template('index.html', result=result, text=text)

def preprocess_text(text):
    
    case_folding(text)
    normalize_text(text)
    stopwords_removal(text)
    stemming_text(text)
    return text

# def perform_sentiment_analysis(text):
#     # Preprocess the input text
#     preprocessed_text = preprocess_text(text)

#     # Tokenize and pad the sequence
#     max_length = 100  # Adjust based on your dataset
#     text_seq = tokenizer.texts_to_sequences([preprocessed_text])
#     text_padded = pad_sequences(text_seq, maxlen=max_length, padding='post')

#     # Make prediction
#     prediction = model.predict(text_padded)
#     sentiment = 'Positive' if prediction > 0.5 else 'Negative'

#     return sentiment

def perform_sentiment_analysis(text):
    # Preprocess the input text
    preprocessed_text = preprocess_text(text)

    # Tokenize and pad the sequence
    max_length = 100  # Adjust based on your dataset
    text_seq = tokenizer.texts_to_sequences([preprocessed_text])
    text_padded = pad_sequences(text_seq, maxlen=max_length, padding='post')

    # Make prediction
    prediction = model.predict(text_padded)

    # Check the sentiment based on the maximum predicted value
    max_prob_index = np.argmax(prediction)
    sentiment = 'Positive' if max_prob_index == 2 else 'Negative'

    return sentiment


if __name__ == '__main__':
    app.run(debug=True)
