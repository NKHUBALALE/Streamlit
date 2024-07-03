# Streamlit dependencies
import streamlit as st
import joblib
import os

# Data dependencies
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Define file paths for models and vectorizer
streamlit_dir = os.path.dirname(__file__)
tfidf_vectorizer_path = os.path.join(streamlit_dir, 'tfidf_vectorizer.pkl')
lr_model_path = os.path.join(streamlit_dir, 'lr_classifier_model.pkl')
nb_model_path = os.path.join(streamlit_dir, 'nb_classifier_model.pkl')
rf_model_path = os.path.join(streamlit_dir, 'rf_classifier_model.pkl')
svm_model_path = os.path.join(streamlit_dir, 'svm_classifier_model.pkl')
knn_model_path = os.path.join(streamlit_dir, 'knn_classifier_model.pkl')
mlp_model_path = os.path.join(streamlit_dir, 'mlp_classifier_model.pkl')
data_path = os.path.join(streamlit_dir, 'train.csv')
wordcloud_path = os.path.join(streamlit_dir, 'wordcloud_by_category.png')

# Load your vectorizer and models
tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
lr_model = joblib.load(lr_model_path)
nb_model = joblib.load(nb_model_path)
rf_model = joblib.load(rf_model_path)
svm_model = joblib.load(svm_model_path)
knn_model = joblib.load(knn_model_path)
mlp_model = joblib.load(mlp_model_path)

# Load your training data
data = pd.read_csv(data_path)

# CSS for green text color
sidebar_style = """
    <style>
        .sidebar-section {
            color: green;
            padding: 10px;
            margin-bottom: 10px;
        }
    </style>
"""

# Function to generate word clouds
def generate_word_clouds(data):
    categories = ['business', 'education', 'entertainment', 'sports', 'technology']
    full_title = ['Business', 'Education', 'Entertainment', 'Sports', 'Technology']
    exclude_words = set(['business', 'education', 'entertainment', 'sports', 'technology', 'https'])
    wc = WordCloud(width=500, height=500, background_color='white', colormap='Dark2', max_font_size=150, random_state=42, stopwords=exclude_words)

    plt.rcParams['figure.figsize'] = [20, 15]
    plt.subplots_adjust(top=0.9)  # Adjust the top of the subplots to shift the headings up

    # Create subplots
    for i, category in enumerate(categories):
        category_text = ' '.join(data.loc[data['category'] == category, 'cleaned_content'])
        wc.generate(category_text)
        
        plt.subplot(2, 3, i + 1)  # 2 rows, 3 columns, and position i+1
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.title(full_title[i])
    
    st.pyplot(plt)

# Main function to build the Streamlit app
def main():
    """News Classifier App with Streamlit """

    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("News Classifier")
    st.subheader("Analyzing news articles")

    # Inject CSS for green text color in the sidebar
    st.sidebar.markdown(sidebar_style, unsafe_allow_html=True)

    # Sidebar navigation tip
    st.sidebar.markdown(
        """
        <div class="sidebar-section">
            <p><b>Navigation Tip:</b></p>
            <p>Click the dropdown menu above to navigate between pages.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Sidebar with selection box for different pages
    options = ["Information", "EDA", "Prediction"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "Information" page
    if selection == "Information":
        st.info("General Information")

        st.markdown(
            """
            The purpose of the classification models is to read articles and classify them into categories, which are sports, education, entertainment, business, and technology.\n\n"
            Model details:\n\n"
            The application makes use of various models which are Multinomial Naive Bayes, Random Forest, KNN, Logistic Regression, Support Vector Machine (SVM), and Multi-Layer Perceptron (MLP) Classifier.
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

        # Plotting category distribution
        category_counts = data['category'].value_counts()

        plt.figure(figsize=(10, 6))
        sns.barplot(x=category_counts.index, y=category_counts.values, palette='viridis')
        plt.xlabel('Category')
        plt.ylabel('Number of Articles')
        plt.title('Distribution of Articles by Category')
        plt.xticks(rotation=45)

        st.pyplot(plt)

        # Button to view balanced data
        if st.button("View Balanced Data"):
            st.markdown(
                """
                To address the imbalance, the dataset was resampled and these are the results:
                """
            )

            # Separate each class
            df_majority = data[data['category'] == 'education']
            df_technology = data[data['category'] == 'technology']
            df_entertainment = data[data['category'] == 'entertainment']
            df_business = data[data['category'] == 'business']
            df_sports = data[data['category'] == 'sports']

            # Determine the target sample size for balancing
            target_size = max(len(df_technology), len(df_entertainment), len(df_business), len(df_sports))

            # Upsample minority classes
            df_technology_upsampled = resample(df_technology,
                                               replace=True,
                                               n_samples=target_size,
                                               random_state=123)

            df_entertainment_upsampled = resample(df_entertainment,
                                                  replace=True,
                                                  n_samples=target_size,
                                                  random_state=123)

            df_business_upsampled = resample(df_business,
                                             replace=True,
                                             n_samples=target_size,
                                             random_state=123)

            df_sports_upsampled = resample(df_sports,
                                           replace=True,
                                           n_samples=target_size,
                                           random_state=123)

            # Undersample the majority class
            df_majority_downsampled = resample(df_majority,
                                               replace=False,
                                               n_samples=target_size,
                                               random_state=123)

            # Combine the resampled dataframes
            df_balanced = pd.concat([df_majority_downsampled,
                                     df_technology_upsampled,
                                     df_entertainment_upsampled,
                                     df_business_upsampled,
                                     df_sports_upsampled])

            # Shuffle the combined dataframe to mix the classes
            df_balanced = df_balanced.sample(frac=1, random_state=123).reset_index(drop=True)

            # Plotting balanced category distribution
            balanced_counts = df_balanced['category'].value_counts()

            plt.figure(figsize=(10, 6))
            sns.barplot(x=balanced_counts.index, y=balanced_counts.values, palette='viridis')
            plt.xlabel('Category')
            plt.ylabel('Number of Articles')
            plt.title('Balanced Distribution of Articles by Category')
            plt.xticks(rotation=45)

            st.pyplot(plt)

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
            "KNN": knn_model,
            "MLP Classifier": mlp_model
        }

        # Navigation tip below the model selection dropdown
        st.sidebar.markdown(
            """
            <div class="sidebar-section">
                <p><b>Navigation Tip:</b></p>
                <p style="color: green;">Click the dropdown under 'Select Model' to navigate between models.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

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

# Execute the main function
if __name__ == '__main__':
    main()
