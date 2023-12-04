import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request

df = pd.read_excel("dataset.xlsx")

import string
import re
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nltk.download('punkt')

def case_folding(sentence):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # Emojis in the first range
        u"\U0001F300-\U0001F5FF"  # Emojis in the second range
        u"\U0001F680-\U0001F6FF"  # Emojis in the third range
        u"\U0001F700-\U0001F77F"  # Emojis in the fourth range
        u"\U0001F780-\U0001F7FF"  # Emojis in the fifth range
        u"\U0001F800-\U0001F8FF"  # Emojis in the sixth range
        u"\U0001F900-\U0001F9FF"  # Emojis in the seventh range
        u"\U0001FA00-\U0001FA6F"  # Emojis in the eighth range
        u"\U0001FA70-\U0001FAFF"  # Emojis in the ninth range
        u"\U0001F004-\U0001F0CF"  # Emojis in the tenth range
        "]+", flags=re.UNICODE)

    sentence = emoji_pattern.sub(r'', sentence)
    sentence = sentence.translate(str.maketrans("","", string.punctuation)).lower()
    sentence = re.sub(r"\d+", "", sentence)
    sentence = sentence.replace("/", " ")
    return sentence

def load_abbreviation_file(file_path):
    try:
        with open(file_path, "r") as file:
            abbreviations = json.load(file)
        return abbreviations
    except FileExistsError:
        print(f"File not found {file_path}")
        return {}

#Reading the abbreviation file path for preprocessing
file_path = "abbreviation_file.txt"
abbreviation_file = load_abbreviation_file(file_path)

def normalize_text(sentence):
    words = sentence.lower().split()
    words_normalized = []
    for word in words:
        for full_form, abbreviations in abbreviation_file.items():
            if word.lower() in abbreviations:
                words_normalized.append(full_form)
                break
        else:
            words_normalized.append(word)
    return " ".join(words_normalized)

def stopwords_removal(sentence):
    tokens = word_tokenize(sentence)
    liststopwords =  set(stopwords.words('indonesian'))

    custom_stopwords_file = "more_stopwords.txt"

    custom_stopwords = set()
    with open(custom_stopwords_file, "r") as file:
        for line in file:
            custom_stopwords.add(line.strip())

    combined_stopwords = liststopwords.union(custom_stopwords)

    with open(custom_stopwords_file, "w") as file:
        for word in combined_stopwords:
            file.write(word + "\n")

def remove_custom_stopwords(sentence, custom_stopwords_file):
    custom_stopwords = set()
    with open(custom_stopwords_file, 'r') as file:
        for line in file:
            custom_stopwords.add(line.strip())

    words = word_tokenize(sentence)

    filtered_words = [word for word in words if word.lower() not in custom_stopwords]

    cleaned_text = ' '.join(filtered_words)

    return cleaned_text

def stemming_text(sentence):
    factory = StemmerFactory()
    Stemmer = factory.create_stemmer()

    sentence = Stemmer.stem(sentence)
    return sentence

df['case_folding'] = df['review_text'].apply(case_folding)
df['normalized_text'] = df['case_folding'].apply(normalize_text)
df['stopword_removed'] = df['normalized_text'].apply(lambda x: remove_custom_stopwords(x, 'more_stopwords.txt'))
df['stemmed_text'] = df['stopword_removed'].apply(stemming_text)
pd.set_option('display.max_colwidth', None)
#df.head(1)

dfnew = df[["stemmed_text", "review_rating"]]
#dfnew.head()

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras import models

# Custom label mapping
label_mapping = {1: 'Negative', 2: 'Negative', 3: 'Neutral', 4: 'Positive', 5: 'Positive'}

# Create a new DataFrame with only the relevant columns
df_encoded = pd.DataFrame()
df_encoded['stemmed_text'] = dfnew['stemmed_text']

# Map labels to the custom categories
df_encoded['sentiment_category'] = dfnew['review_rating'].map(label_mapping)

# Handle any remaining NaN values (replace with a default value, e.g., 'Neutral')
df_encoded['sentiment_category'].fillna('Neutral', inplace=True)

# Now you can proceed with label encoding
label_encoder = LabelEncoder()
df_encoded['encoded_label'] = label_encoder.fit_transform(df_encoded['sentiment_category'])

# Print the new DataFrame to inspect
# df_encoded.head()

X_train, X_test, y_train, y_test = train_test_split(df_encoded['stemmed_text'], df_encoded['encoded_label'], test_size=0.2, random_state=42)

max_words = 10000
tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_length = 100  # Adjust based on your dataset
X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

# Assuming you have a DataFrame df_encoded with 'text' and 'encoded_label' columns
# One-hot encode the labels
df_encoded['encoded_label'] = to_categorical(df_encoded['encoded_label'])
#df_encoded.head()

# Build a simple neural network
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_length))
model.add(Flatten())
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=3, activation='softmax'))  # 3 output units for three categories

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_padded, to_categorical(y_train), epochs=15, validation_data=(X_test_padded, to_categorical(y_test)))

#Save the model and tokenizer
model.save('sentiment_model.h5')
tokenizer_config = tokenizer.get_config()
tokenizer_config['num_words'] = max_words
with open('tokenizer_config.json', 'w') as config_file:
    config_file.write(json.dumps(tokenizer_config))

from tensorflow.keras.preprocessing.text import text_to_word_sequence

# ...

# Load the model and tokenizer
model = models.load_model('sentiment_model.h5')
with open('tokenizer_config.json') as config_file:
    config = json.load(config_file)
    max_words = config['num_words']
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.word_index = json.loads(config['word_index'])
    tokenizer.index_word = {str(i): word for word, i in tokenizer.word_index.items()}

# ...

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        result = perform_sentiment_analysis(text)
        return render_template('index.html', result=result, text=text)

def perform_sentiment_analysis(text):
    # Preprocess the input text
    preprocessed_text = preprocess_text(text)

    # Tokenize and pad the sequence
    max_length = 100  # Adjust based on your dataset
    text_seq = tokenizer.texts_to_sequences([preprocessed_text])
    text_padded = pad_sequences(text_seq, maxlen=max_length, padding='post')

    # Make prediction
    prediction = model.predict(text_padded)
    sentiment = 'Positive' if prediction > 0.5 else 'Negative'

    return sentiment

if __name__ == '__main__':
    app.run(debug=True)