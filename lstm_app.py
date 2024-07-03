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
    'Education and Training': 'LSTM_models/education/L2_education_model.h5',
    "Finance and Accounting" : 'LSTM_models/finance/L2_finance_model.h5', 
    "Fleet and Transportation Management" :'LSTM_models/transport/L2_transport_model.h5',
    "Greeting_l1": 'LSTM_models/greeting/L2_greeting_model.h5',
    "HR": 'LSTM_models/hr/L2_hr_model.h5',
    "Healthcare and Medical":'LSTM_models/health/L2_health_model.h5',
    "Hospitality and Travel": 'LSTM_models/travel/L2_travel_model.h5',
    "IT and Communications": 'LSTM_models/it/L2_it_model.h5',
    "Legal and Risk Management": 'LSTM_models/legal/L2_legal_model.h5',
    "Manufacturing and Supply Chain": 'LSTM_models/scm/L2_scm_model.h5',
    "Marketing and Sales": 'LSTM_models/marketing/L2_marketing_model.h5',
    "Nonprofit and Social Services": 'LSTM_models/nonprofit/L2_nonprofit_model.h5',
    "Project and Operations Management" : 'LSTM_models/pm/L2_pm_model.h5',
    "Retail and Ecommerce": 'LSTM_models/retail/L2_retail_model.h5'
}


secondary_tokenizer_paths = {
    'Business Intelligence and Analytics': 'LSTM_models/bi/L2_bi_tokenizer.pkl',
    'Construction and Real Estate Development': 'LSTM_models/construction/L2_construction_tokenizer.pkl',
    'Customer Relationship Management': 'LSTM_models/crm/L2_crm_tokenizer.pkl',
    'Education and Training': 'LSTM_models/education/L2_education_tokenizer.pkl',
    "Finance and Accounting" : 'LSTM_models/finance/L2_finance_tokenizer.pkl', 
    "Fleet and Transportation Management" :'LSTM_models/transport/L2_transport_tokenizer.pkl',
    "Greeting_l1": 'LSTM_models/greeting/L2_greeting_tokenizer.pkl',
    "HR": 'LSTM_models/hr/L2_hr_tokenizer.pkl',
    "Healthcare and Medical":'LSTM_models/health/L2_health_tokenizer.pkl',
    "Hospitality and Travel": 'LSTM_models/travel/L2_travel_tokenizer.pkl',
    "IT and Communications": 'LSTM_models/it/L2_it_tokenizer.pkl',
    "Legal and Risk Management": 'LSTM_models/legal/L2_legal_tokenizer.pkl',
    "Manufacturing and Supply Chain": 'LSTM_models/scm/L2_scm_tokenizer.pkl',
    "Marketing and Sales": 'LSTM_models/marketing/L2_marketing_tokenizer.pkl',
    "Nonprofit and Social Services": 'LSTM_models/nonprofit/L2_nonprofit_tokenizer.pkl',
    "Project and Operations Management" : 'LSTM_models/pm/L2_pm_tokenizer.pkl',
    "Retail and Ecommerce": 'LSTM_models/retail/L2_retail_tokenizer.pkl'
}

secondary_label_paths = {
    'Business Intelligence and Analytics': 'LSTM_models/bi/L2_bi_dct.json',
    'Construction and Real Estate Development': 'LSTM_models/construction/L2_construction_dct.json',
    'Customer Relationship Management': 'LSTM_models/crm/L2_crm_dct.json',
    'Education and Training': 'LSTM_models/education/L2_education_dct.json',
    "Finance and Accounting" : 'LSTM_models/finance/L2_finance_dct.json', 
    "Fleet and Transportation Management" :'LSTM_models/transport/L2_transport_dct.json',
    "Greeting_l1": 'LSTM_models/greeting/L2_greeting_dct.json',
    "HR": 'LSTM_models/hr/L2_hr_dct.json',
    "Healthcare and Medical":'LSTM_models/health/L2_health_dct.json',
    "Hospitality and Travel": 'LSTM_models/travel/L2_travel_dct.json',
    "IT and Communications": 'LSTM_models/it/L2_it_dct.json',
    "Legal and Risk Management": 'LSTM_models/legal/L2_legal_dct.json',
    "Manufacturing and Supply Chain": 'LSTM_models/scm/L2_scm_dct.json',
    "Marketing and Sales": 'LSTM_models/marketing/L2_marketing_dct.json',
    "Nonprofit and Social Services": 'LSTM_models/nonprofit/L2_nonprofit_dct.json',
    "Project and Operations Management" : 'LSTM_models/pm/L2_pm_dct.json',
    "Retail and Ecommerce": 'LSTM_models/retail/L2_retail_dct.json'

# Initialize the classifier
classifier = HierarchicalTextClassifier(main_model_path, main_tokenizer_path, main_label_path,
                                        secondary_model_paths, secondary_tokenizer_paths, secondary_label_paths)

st.title('Hierarchical Text Classifier')

# Sidebar menu
st.sidebar.title('Catgories')
st.sidebar.markdown('write a question from any of the below category to classify the text:')

categories = list(secondary_model_paths.keys())
selected_category = st.sidebar.radio('Categories', categories)

if st.sidebar.button("About Us"):
    st.sidebar.markdown("""
    This is a Streamlit app demonstrating hierarchical text classification using deep learning models.
    """)

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
