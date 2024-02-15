# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 03:33:30 2024
@author: DELL
"""

## Streamlit Application for Sentiment Analysis

import streamlit as st
import pickle
import pandas as pd

# Importing the Data
df = pd.read_excel("C:\\Users\\DELL\\Downloads\\News_Classifier_org.xlsx")

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
# Function to clean and preprocess the text
def clean_and_preprocess(text):
    # Remove HTML tags using BeautifulSoup
    soup = BeautifulSoup(text, "html.parser")
    clean_text = soup.get_text(separator=" ")

    # Remove non-alphanumeric characters and extra whitespaces
    clean_text = re.sub(r'[^a-zA-Z\s]', '', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    # Tokenization and removing stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(clean_text)
    filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Join the tokens back into a string
    preprocessed_text = ' '.join(lemmatized_tokens)

    return preprocessed_text

# Streamlit app
st.title("News Classifier App")

# Text input for user query
user_input = st.text_input("Enter a news headline:")

# Load the TF-IDF vectorizer and KNN model, pca, preprocess from pickle files

with open("C:\\Users\\DELL\\Downloads\\News_Classifier\\clean.pkl", "rb") as f:
    clean = pickle.load(f)

with open("C:\\Users\\DELL\\Downloads\\News_Classifier\\vec.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("C:\\Users\\DELL\\Downloads\\News_Classifier\\knn_Model_S.pkl", "rb") as f:
    knn = pickle.load(f)

with open("C:\\Users\\DELL\\Downloads\\News_Classifier\\pca.pkl", "rb") as f:
    pca = pickle.load(f)

# Predict cluster label for user input
if user_input:
    
    # Clean and preprocess user input
    user_input = clean(user_input)
    
    # Transform user input
    user_input_vectorized = vectorizer.transform([user_input])

    # Apply PCA transformation
    user_input_transformed = pca.transform(user_input_vectorized)

    # Predict cluster label
    predicted_label = knn.predict(user_input_transformed)[0]
    News_headline_label = {1: 'Sports News', 0: 'Weather Forecast', 2: 'Politics',3: 'International News'}[predicted_label]

    st.write("News Classifier:", News_headline_label)
