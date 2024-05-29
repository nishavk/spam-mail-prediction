import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the TF-IDF vectorizer
with open(r'C:/Users/ELCOT/Desktop/spam mail prediction/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Load the spam mail detection model
spam_mail_model = pickle.load(open(r'C:/Users/ELCOT/Desktop/spam mail prediction/spam_detection_model.sav', 'rb'))

# Set page title and icon
st.set_page_config(page_title="Spam Mail Prediction", page_icon="📧")

# Sidebar for navigation
with st.sidebar:
    st.title("Navigation")
    selected = st.radio("Go to:", ["Home", "Spam Mail Prediction"])

if selected == "Home":
    st.title("Welcome to the Spam Mail Prediction App")
    st.write("Use the navigation on the left to access the spam mail prediction feature.")

elif selected == "Spam Mail Prediction":
    st.title('Spam Mail Prediction App')
    st.write("Enter the text of the email below to predict whether it's spam or not.")

    # Create a text area for user input
    user_input = st.text_area("Email Text", "")

    # Code for Prediction
    spam_diagnosis = ''

    # Creating a button for Prediction
    if st.button('Predict'):
        # Ensure the user provided some input
        if user_input:
            # Transform user input into TF-IDF features
            input_data_features = tfidf_vectorizer.transform([user_input])

            # Make the prediction
            prediction = spam_mail_model.predict(input_data_features)

            if prediction[0] == 1:
                spam_diagnosis = 'This email is classified as ham.'
            else:
                spam_diagnosis = 'This email is spam.'

    # Display the prediction result
    st.write('')
    st.markdown('## Prediction Result:')
    st.info(spam_diagnosis)
