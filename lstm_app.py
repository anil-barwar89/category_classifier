import streamlit as st
import re, string, pickle , os, json, nltk
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

from config import config

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

nltk.download('stopwords')
nltk.download('punkt') 
nltk.download('wordnet')

class HierarchicalTextClassifier:
    def __init__(self, config,  maxlen=50):
        self.config = config
        self.maxlen = maxlen

    def load_main_model(self):
        # Load the main model, tokenizer, and labels
        main_model = self.load_model(self.config["main_model_path"])
        main_tokenizer = self.load_tokenizer(self.config['main_tokenizer_path'])
        main_labels = self.load_labels(self.config['main_label_path'])
        return main_model, main_tokenizer, main_labels
        
    def load_secondary_model(self, category):
        # Load the secondary model, tokenizer, and labels based on category
        secondary_model = self.load_model(self.config["secondary_model_paths"].get(category))
        secondary_tokenizer = self.load_tokenizer(self.config["secondary_tokenizer_paths"].get(category))
        secondary_labels = self.load_labels(self.config["secondary_label_paths"].get(category))
        return secondary_model, secondary_tokenizer, secondary_labels

    def load_model(self, path):
        return load_model(path)  

    def load_tokenizer(self, path):
        with open(path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        return tokenizer
            
    def load_labels(self, path):
        with open(path, 'rb') as handle:
            labels = json.load(handle)
        return labels

    @staticmethod
    def preprocessing_text(text):
        text = text.lower()
        translation_table = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        text = text.translate(translation_table)
        text = ' '.join(text.split())
        text = re.sub(r'\s*\b\d+\b\s*', ' ', text)
        text = ' '.join(text.split())
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [lemmatizer.lemmatize(word, pos='v') for word in tokens]
        return " ".join(tokens)
        
    def model_prediction(self, model, tokenizer, label,  text):
        # tokenizing and padding the processed input 
        sequences = tokenizer.texts_to_sequences([text])
        padded_sequences = pad_sequences(sequences, maxlen=self.maxlen, padding='post')
        # model prediction
        pred_prob = model.predict(padded_sequences)       
        pred_class = np.argmax(pred_prob, axis=1)[0]      
        return pred_prob[0][pred_class], label[str(pred_class)]
        
    def classify_text(self, text):
        # Preprocess the text
        processed_text = self.preprocessing_text(text)
        # load primary(L1) model, tokenizer, labels
        main_model, main_tokenizer, main_labels = self.load_main_model()
        # predict L1 class, L1 probability from primary model
        main_pred_prob, main_pred_class = self.model_prediction( main_model, main_tokenizer, main_labels, processed_text )
        
        # Secondary classification if a secondary model exists for the main class
        if main_pred_class in self.config['secondary_label_paths']:
            # load secondary(L2) model, tokenizer, labels on predicted L1 class label
            secondary_model, secondary_tokenizer, secondary_labels = self.load_secondary_model(main_pred_class)
             # predict L1 class, L1 probability from primary model
            secondary_pred_prob, secondary_pred_class = self.model_prediction(secondary_model, secondary_tokenizer, secondary_labels, processed_text)
            return main_pred_class, main_pred_prob, secondary_pred_class, secondary_pred_prob
        else:
            return main_pred_class, main_pred_prob, None, None

# Initialize the classifier
classifier = HierarchicalTextClassifier(config)

st.title('Hierarchical Text Classifier')

# Sidebar menu
# st.sidebar.title('Catgories')
# st.sidebar.markdown('write a question from any of the below category to classify the text:')

with open(config['l1_l2_mapping'], 'r') as file:
    categories = json.load(file)

# Sidebar menu
st.sidebar.title('Categories')
selected_category = st.sidebar.radio('Select a Category', list(categories.keys()))

# Display subcategories based on selected category
if selected_category:
    st.sidebar.markdown(f'Subcategories of {selected_category}:')
    subcategories = categories[selected_category]
    selected_subcategory = st.sidebar.selectbox('', subcategories)

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
