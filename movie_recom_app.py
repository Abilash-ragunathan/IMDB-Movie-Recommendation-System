import streamlit as st
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load Dataset

df = pd.read_csv("D:\obito\obito\mini5\imdb_movies_2024.csv")


# Text Cleaning Function

def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove punctuation
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    return text


# Preprocess Storylines

df['clean_storyline'] = df['Storyline'].fillna('').apply(clean_text)


# TF-IDF Vectorization

tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(df['clean_storyline'])


# Streamlit UI

st.set_page_config(
    page_title="Movie Recommendation System"
)

st.title("Movie Recommendation System")

st.write(
    "Enter a movie storyline and get the Top 5 similar movie recommendations."
)

user_story = st.text_area(
    "Enter Movie Storyline",
    height=150
)


# Recommendation Button

if st.button("Recommend Movies"):

    if user_story.strip() == "":
        st.warning("Please enter a movie storyline.")
    else:

        # Clean user input
        cleaned_story = clean_text(user_story)

        # Convert user input into vector
        user_vector = tfidf.transform([cleaned_story])

        # Calculate similarity
        similarity_scores = cosine_similarity(
            user_vector,
            tfidf_matrix
        ).flatten()

        # Get Top 5 Recommendations
        top_indices = similarity_scores.argsort()[-5:][::-1]

        st.subheader("Top 5 Recommended Movies")

        for rank, idx in enumerate(top_indices, start=1):

            movie_name = df.iloc[idx]['Movie Name']
            storyline = df.iloc[idx]['Storyline']
            score = similarity_scores[idx]

            st.markdown(f"## {rank}. {movie_name}")

            st.write("**Storyline:**")
            st.write(storyline)

            # st.write(
            #     f"**Similarity Score:** {score:.4f}"
            # )

            # st.divider()