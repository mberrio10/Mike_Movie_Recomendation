import streamlit as st
import pickle
import joblib
import pandas as pd

# Load precomputed files
with open("models/movie_features.pkl", "rb") as f:
    movieFeatures = pickle.load(f)

with open("models/knn_model.pkl", "rb") as f:
    knn_model = joblib.load(f)  # Load trained KNN model

data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/gperdrizet-k-nearest-neighbors/refs/heads/main/data/raw/tmdb_5000_movies.csv")  # Load movie metadata

# Function to get recommendations
def get_movie_recommendations(input_movie, knn_model, movieFeatures, data, top_n=5):
    if input_movie not in data['original_title'].values:
        return ["Movie not found"]
    
    movie_index = data[data['original_title'] == input_movie].index[0]
    distances, indices = knn_model.kneighbors([movieFeatures[movie_index]])

    recommendations = [(data.iloc[idx]['original_title'], distances[0][i]) for i, idx in enumerate(indices[0])]
    return recommendations

# Streamlit UI
st.title("🎬 Movie Recommendation System")
input_movie = st.text_input("Enter a movie name:", "Avatar")

if st.button("Get Recommendations"):
    recommendations = get_movie_recommendations(input_movie, knn_model, movieFeatures, data)
    
    st.write(f"Recommendations for **{input_movie}**:")
    for movie, distance in recommendations:
        st.write(f"- {movie} (Similarity Score: {distance:.2f})")