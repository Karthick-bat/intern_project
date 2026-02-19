import streamlit as st
from recommend import load_data, preprocess_data, collaborative_recommendations, content_based_recommendations

st.title("🎬 Movie Recommendation System")

# Load and preprocess data
movies = load_data()
data, user_movie_matrix = preprocess_data(movies, ratings)

menu = st.radio("Choose Recommendation Type:", ("Collaborative Filtering", "Content-Based Filtering"))

if menu == "Collaborative Filtering":
    max_user = int(ratings['userId'].max())
    user_id = st.slider("Select your User ID", 1, max_user, 1)

    
    if st.button("Get Recommendations"):
        recs = collaborative_recommendations(user_id, user_movie_matrix)
        st.subheader("🎉 Top 5 Recommendations")
        for movie in recs:
            st.write("👉", movie)

else:
    genre = st.text_input("Enter your favorite genre (e.g. Comedy, Action, Drama)")
    
    if st.button("Recommend"):
        recs = content_based_recommendations(movies, genre)
        st.subheader("🎬 Top 5 Genre-Based Recommendations")
        for movie in recs:
            st.write("👉", movie)
