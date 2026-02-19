import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    return movies, ratings

def preprocess_data(movies, ratings):
    # Merge datasets
    data = pd.merge(ratings, movies, on='movieId')
    
    # Create pivot table for collaborative filtering
    user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating')
    return data, user_movie_matrix

def collaborative_recommendations(user_id, user_movie_matrix):
    user_similarity = cosine_similarity(user_movie_matrix.fillna(0))
    sim_scores = list(enumerate(user_similarity[user_id - 1]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    similar_users = [i[0] for i in sim_scores[1:4]]  # Top 3 similar users

    movie_scores = user_movie_matrix.iloc[similar_users].mean().sort_values(ascending=False)
    return movie_scores.head(5).index.tolist()

def content_based_recommendations(movies, fav_genre):
    movies['genres'] = movies['genres'].replace('(no genres listed)', '')
    tfidf = TfidfVectorizer(token_pattern='[a-zA-Z\-]+')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])

    user_pref = tfidf.transform([fav_genre])
    cosine_sim = cosine_similarity(user_pref, tfidf_matrix)
    
    top_indices = cosine_sim[0].argsort()[-5:][::-1]
    return movies['title'].iloc[top_indices].tolist()
