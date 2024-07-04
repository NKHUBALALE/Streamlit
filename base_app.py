# Streamlit dependencies
import streamlit as st
import joblib
import os

# Data dependencies
import pandas as pd
from PIL import Image

# Define file paths for models and vectorizer
streamlit_dir = os.path.dirname(__file__)
tfidf_vectorizer_path = os.path.join(streamlit_dir, 'tfidf_vectorizer.pkl')
lr_model_path = os.path.join(streamlit_dir, 'lr_classifier_model.pkl')
nb_model_path = os.path.join(streamlit_dir, 'nb_classifier_model.pkl')
rf_model_path = os.path.join(streamlit_dir, 'rf_classifier_model.pkl')
svm_model_path = os.path.join(streamlit_dir, 'svm_classifier_model.pkl')
data_path = os.path.join(streamlit_dir, 'train.csv')
wordcloud_path = os.path.join(streamlit_dir, 'wordcloud_by_category.png')
imbalanced_distribution_path = os.path.join(streamlit_dir, 'imbalanced_distribution.png')
balanced_class_category_path = os.path.join(streamlit_dir, 'balanced_class_category.png')
stone_image_path = os.path.join(streamlit_dir, 'stone.png')
screenshot_image_path = os.path.join(streamlit_dir, 'Screenshot 2024-07-04 203832.png')

# Load your vectorizer and models
tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
lr_model = joblib.load(lr_model_path)
nb_model = joblib.load(nb_model_path)
rf_model = joblib.load(rf_model_path)
svm_model = joblib.load(svm_model_path)

# Load your training data
data = pd.read_csv(data_path)

# Main function to build the Streamlit app
def main():
    """News Classifier App with Streamlit """

    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("News Classifier")
    st.subheader("Analyzing news articles")

    # Sidebar navigation tip in green
    st.sidebar.markdown(
        """
        <div class="sidebar-section" style="background-color:#28a745;padding:10px;border-radius:10px;">
            <p style="color:white;"><b>Navigation Tip:</b></p>
            <p style="color:white;">Click the dropdown menu above to navigate between pages.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Sidebar with selection box for different pages
    options = ["Home", "Information", "EDA", "Prediction", "Feedback", "About Us"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "Home" page
    if selection == "Home":
        st.info("Welcome to the News Classifier App!")
        
        st.markdown(
            """
            This application is designed to classify news articles into categories such as sports, education, entertainment, business, and technology using various machine learning models.
            """
        )
        
        st.markdown(
            """
            Use the sidebar to navigate to different sections of the application, including Information, EDA (Exploratory Data Analysis), Prediction, Feedback, and About Us.
            """
        )
        
        # Displaying the screenshot image below the texts
        screenshot_image = Image.open(screenshot_image_path)
        st.image(screenshot_image, caption='Welcome to the News Classifier App!')

    # Building out the "Information" page
    if selection == "Information":
        st.info("General Information")

        st.markdown(
            """
            The purpose of the classification models is to read articles and classify them into categories, which are sports, education, entertainment, business, and technology.
            
            ### Model Details:
            
            The application makes use of various models which are Multinomial Naive Bayes, Random Forest, Logistic Regression, and Support Vector Machine (SVM).
            """
        )
        
        # Adding Documentation section
        st.markdown(
            """
            ### Documentation:
            
            - **Multinomial Naive Bayes:** A probabilistic model based on Bayes' theorem, used for text classification.
            - **Random Forest:** An ensemble learning method using multiple decision trees for improved accuracy.
            - **Logistic Regression:** A statistical model for binary classification, extended for multi-class classification.
            - **Support Vector Machine (SVM):** A supervised learning model for classification by finding the optimal hyperplane.
            """
        )

    # Building out the "EDA" page
    if selection == "EDA":
        st.info("Exploratory Data Analysis")

        st.markdown(
            """
            Overview and Introduction:

            - The purpose of the Exploratory Data Analysis (EDA) is to examine how news articles are distributed among different categories: sports, education, entertainment, business, and technology.
            - The dataset used for this analysis includes labeled articles that train our models to classify text into these categories.
            - Understanding the distribution of data helps in identifying patterns and potential biases, which are critical for developing accurate classification models.
            """
        )

        # Displaying the imbalanced distribution image
        st.subheader("Imbalanced Class Distribution")
        imbalanced_image = Image.open(imbalanced_distribution_path)
        st.image(imbalanced_image, caption='Imbalanced Distribution of Articles by Category')

        # Button to view balanced data
        if st.button("View Balanced Data"):
            st.markdown(
                """
                To address the imbalance, the dataset was resampled and these are the results:
                """
            )

            # Displaying the balanced class category image
            balanced_image = Image.open(balanced_class_category_path)
            st.image(balanced_image, caption='Balanced Distribution of Articles by Category')

        # Displaying word cloud image
        st.subheader("The Most Used Words by Category")
        wordcloud_image = Image.open(wordcloud_path)
        st.image(wordcloud_image, caption='Word Cloud by Category')

    # Building out the prediction page
    if selection == "Prediction":
        st.info("Prediction with ML Models")

        # Sidebar to select model
        model_options = {
            "Logistic Regression": lr_model,
            "Naive Bayes": nb_model,
            "Random Forest": rf_model,
            "SVM": svm_model,
        }

        selected_model = st.sidebar.radio("Select Model", list(model_options.keys()))

        # Store selected model in session state
        st.session_state.selected_model = selected_model

        # Display the selected model in green
        for model_name in model_options:
            if st.session_state.selected_model == model_name:
                st.sidebar.markdown(f'<p style="color:green;">{model_name}</p>', unsafe_allow_html=True)
            else:
                st.sidebar.markdown(f'<p>{model_name}</p>', unsafe_allow_html=True)

        # Creating a text box for user input
        news_text = st.text_area("Type a text to classify", "")

        if st.button("Classify"):
            # Transforming user input with vectorizer
            vect_text = tfidf_vectorizer.transform([news_text]).toarray()

            # Make prediction with the selected model
            predictor = model_options[selected_model]
            prediction = predictor.predict(vect_text)[0]

            # Display prediction
            st.success(f"Text Categorized as: {prediction}")

    # Building out the Feedback page
    if selection == "Feedback":
        st.info("Feedback")

        st.markdown(
            """
            We value your feedback! Please provide your comments and suggestions to help us improve this application.
            """
        )

        feedback_text = st.text_area("Your Feedback", "")
        if st.button("Submit Feedback"):
            st.success("Thank you for your feedback!")

    # Building out the About Us page
    if selection == "About Us":
        st.info("About Us")

        # Displaying the stone background image with reduced size
        stone_image = Image.open(stone_image_path)
        st.image(stone_image, use_column_width=False, width=300)

        st.markdown(
            """
            This application was developed by Team_MM1, a group of data science students from the ExploreAI academy under the supervised classification sprint [Team_MM1] as a project to classify news articles into categories using machine learning models. It demonstrates the use of various classification algorithms to analyze and categorize text data.

            For more information or inquiries, please contact us at mm1_classification@sandtech.co.za.

            ---

            **Our Mission:**

            "We are here to innovate Africa and the world through data-driven insights, one article at a time."
            """
        )

# Execute the main function
if __name__ == '__main__':
    main()
