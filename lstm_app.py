import streamlit as st
import re, string
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import json
import nltk

nltk.download('stopwords')
nltk.download('punkt') 
nltk.download('wordnet')

class HierarchicalTextClassifier:
    def __init__(self, main_model_path, main_tokenizer_path, main_label_path,
                 secondary_model_paths, secondary_tokenizer_paths, secondary_label_paths, maxlen=50):
        # Load main level model and tokenizer
        self.main_model = load_model(main_model_path)
        self.maxlen = maxlen

        with open(main_tokenizer_path, 'rb') as handle:
            self.main_tokenizer = pickle.load(handle)

        with open(main_label_path, 'r') as handle:
            self.main_label = json.load(handle)

        # Load secondary level models and tokenizers
        self.secondary_models = {}
        self.secondary_tokenizers = {}
        self.secondary_labels = {}

        for class_label, model_path in secondary_model_paths.items():
            self.secondary_models[class_label] = load_model(model_path)

        for class_label, tokenizer_path in secondary_tokenizer_paths.items():
            with open(tokenizer_path, 'rb') as handle:
                self.secondary_tokenizers[class_label] = pickle.load(handle)

        for class_label, label_path in secondary_label_paths.items():
            with open(label_path, 'r') as handle:
                self.secondary_labels[class_label] = json.load(handle)

    @staticmethod
    def preprocessing_text(text):
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'®', '', text)
        translation_table = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        text = text.translate(translation_table)
        text = ' '.join(text.split())
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'\s*\b\d+\b\s*', ' ', text)
        text = ' '.join(text.split())
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word, pos='v') for word in tokens]
        return " ".join(tokens)

    def classify_text(self, text):
        # Preprocess the text
        processed_text = self.preprocessing_text(text)

        # Primary classification
        sequences = self.main_tokenizer.texts_to_sequences([processed_text])
        padded_sequences = pad_sequences(sequences, maxlen=self.maxlen, padding='post')
        main_pred_prob = self.main_model.predict(padded_sequences)
        main_pred_class = np.argmax(main_pred_prob, axis=1)[0]

        main_pred_probability = main_pred_prob[0][main_pred_class]
        main_pred_label = self.main_label[str(main_pred_class)]

        # Secondary classification if a secondary model exists for the main class
        if main_pred_label in self.secondary_models:
            secondary_tokenizer = self.secondary_tokenizers[main_pred_label]
            sequences = secondary_tokenizer.texts_to_sequences([processed_text])
            padded_sequences = pad_sequences(sequences, maxlen=self.maxlen, padding='post')
            secondary_model = self.secondary_models[main_pred_label]
            secondary_pred_prob = secondary_model.predict(padded_sequences)
            secondary_pred_class = np.argmax(secondary_pred_prob, axis=1)[0]

            secondary_pred_probability = secondary_pred_prob[0][secondary_pred_class]
            secondary_label = self.secondary_labels[main_pred_label]
            secondary_pred_label = secondary_label[str(secondary_pred_class)]

            return main_pred_label, main_pred_probability, secondary_pred_label, secondary_pred_probability
        else:
            return main_pred_label, main_pred_probability, None, None

# Paths to models and tokenizers
main_model_path = 'LSTM_models/L1/L1_model.h5'
main_tokenizer_path = 'LSTM_models/L1/L1_tokenizer.pkl'
main_label_path = 'LSTM_models/L1/L1_dct.json'

secondary_model_paths = {
    'Business Intelligence and Analytics': 'LSTM_models/bi/L2_bi_model.h5',
    'Construction and Real Estate Development': 'LSTM_models/construction/L2_construction_model.h5',
    'Customer Relationship Management': 'LSTM_models/crm/L2_crm_model.h5',
    'Education and Training': 'LSTM_models/education/L2_education_model.h5'
}

secondary_tokenizer_paths = {
    'Business Intelligence and Analytics': 'LSTM_models/bi/L2_bi_tokenizer.pkl',
    'Construction and Real Estate Development': 'LSTM_models/construction/L2_construction_tokenizer.pkl',
    'Customer Relationship Management': 'LSTM_models/crm/L2_crm_tokenizer.pkl',
    'Education and Training': 'LSTM_models/education/L2_education_tokenizer.pkl'
}

secondary_label_paths = {
    'Business Intelligence and Analytics': 'LSTM_models/bi/L2_bi_dct.json',
    'Construction and Real Estate Development': 'LSTM_models/construction/L2_construction_dct.json',
    'Customer Relationship Management': 'LSTM_models/crm/L2_crm_dct.json',
    'Education and Training': 'LSTM_models/education/L2_education_dct.json'
}

# Initialize the classifier
classifier = HierarchicalTextClassifier(main_model_path, main_tokenizer_path, main_label_path,
                                        secondary_model_paths, secondary_tokenizer_paths, secondary_label_paths)

st.title('Hierarchical Text Classifier')
st.write("Enter text for classification:")

text_input = st.text_area("Input text")

if st.button("Classify"):
    if text_input:
        main_class, main_prob, secondary_class, secondary_prob = classifier.classify_text(text_input)

        st.write(f"Main Class: {main_class}")
        st.write(f"Main Class Probability: {main_prob}")

        if secondary_class is not None:
            st.write(f"Secondary Class: {secondary_class}")
            st.write(f"Secondary Class Probability: {secondary_prob}")
        else:
            st.write("No secondary classification available.")
    else:
        st.write("Please enter some text to classify.")
