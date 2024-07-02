"""
Simple Streamlit webserver application for serving developed classification
models.

Author: ExploreAI Academy.

Note:
---------------------------------------------------------------------
Please follow the instructions provided within the README.md file
located within this directory for guidance on how to use this script
correctly.
---------------------------------------------------------------------

Description: This file is used to launch a minimal streamlit web
application. You are expected to extend the functionality of this script
as part of your predict project.

For further help with the Streamlit framework, see:

https://docs.streamlit.io/en/latest/
"""

# Streamlit dependencies
import streamlit as st
import joblib
import os

# Data dependencies
import pandas as pd

# Define file paths for models and vectorizer
streamlit_dir = os.path.dirname(__file__)
tfidf_vectorizer_path = os.path.join(streamlit_dir, 'tfidf_vectorizer.pkl')
lr_model_path = os.path.join(streamlit_dir, 'lr_classifier_model.pkl')
nb_model_path = os.path.join(streamlit_dir, 'nb_classifier_model.pkl')
rf_model_path = os.path.join(streamlit_dir, 'rf_classifier_model.pkl')
svm_model_path = os.path.join(streamlit_dir, 'svm_classifier_model.pkl')
knn_model_path = os.path.join(streamlit_dir, 'knn_classifier_model.pkl')
mlp_model_path = os.path.join(streamlit_dir, 'mlp_classifier_model.pkl')

# Load your vectorizer and models
tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
lr_model = joblib.load(lr_model_path)
nb_model = joblib.load(nb_model_path)
rf_model = joblib.load(rf_model_path)
svm_model = joblib.load(svm_model_path)
knn_model = joblib.load(knn_model_path)
mlp_model = joblib.load(mlp_model_path)

# The main function where we will build the actual app
def main():
    """News Classifier App with Streamlit """

    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("News Classifier")
    st.subheader("Analyzing news articles")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Information", "Prediction"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "Information" page
    if selection == "Information":
        st.info("General Information")
        st.markdown(
            "The purpose of the classification models is to read articles and classify them into categories, which are sports, education, entertainment, business, and technology.\n\n"
            "Model details:\n\n"
            "The application makes use of various models which are Multinomial Naive Bayes, Random Forest, KNN, Logistic Regression, Support Vector Machine (SVM), and Multi-Layer Perceptron (MLP) Classifier."
        )

    # Building out the prediction page
    if selection == "Prediction":
        st.info("Prediction with ML Models")
        
        # Prompt to start prediction
        st.sidebar.info("Select 'Prediction' from the dropdown to start classifying news articles.")

        # Sidebar to select model
        model_options = {
            "Logistic Regression": lr_model,
            "Naive Bayes": nb_model,
            "Random Forest": rf_model,
            "SVM": svm_model,
            "KNN": knn_model,
            "MLP Classifier": mlp_model
        }
        selected_model = st.sidebar.selectbox("Select Model", list(model_options.keys()))

        # Creating a text box for user input
        news_text = st.text_area("Enter Text", "Type Here")

        if st.button("Classify"):
            # Transforming user input with vectorizer
            vect_text = tfidf_vectorizer.transform([news_text]).toarray()

            # Make prediction with the selected model
            predictor = model_options[selected_model]
            prediction = predictor.predict(vect_text)[0]

            # Display prediction
            st.success(f"Text Categorized as: {prediction}")

# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()
