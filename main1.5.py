## CSC 3400 Artifical Intelligence
## Assignment 3: Recommenders

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
users_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('u.user', sep='|', names=users_cols, encoding='latin-1')

ratings_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('u.data', sep='\t', names=ratings_cols, encoding='latin-1')

genre_cols = [
    "genre_unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]
movies_cols = ['movie_id', 'title', 'release_date', "video_release_date", "imdb_url"] + genre_cols
movies = pd.read_csv('u.item', sep='|', names=movies_cols, encoding='latin-1')

# Adjust user and movie IDs to start from 0
ratings['user_id'] = ratings['user_id'] - 1
ratings['movie_id'] = ratings['movie_id'] - 1

# Create a user-movie matrix
user_movie_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

# Perform KMeans clustering
n_clusters = 10  # Number of clusters you want
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(user_movie_matrix)

# Add the cluster label to each user
user_movie_matrix['cluster'] = kmeans.labels_


# Function to get recommendations for a user
def get_movie_recommendations(user_id):
    user_cluster = user_movie_matrix['cluster'].iloc[user_id]

    # Get all users in the same cluster
    similar_users = user_movie_matrix[user_movie_matrix['cluster'] == user_cluster].drop(columns='cluster')

    # Find movies that the input user hasn't watched
    user_movies = set(ratings[ratings['user_id'] == user_id]['movie_id'])
    recommendations = {}

    for similar_user_id in similar_users.index:
        if similar_user_id == user_id:
            continue
        for movie_id in similar_users.columns:
            if movie_id not in user_movies and similar_users.loc[similar_user_id, movie_id] > 0:
                recommendations[movie_id] = recommendations.get(movie_id, 0) + similar_users.loc[
                    similar_user_id, movie_id]

    # Get top 10 movie recommendations
    recommended_movies = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:10]
    return [movie_id for movie_id, score in recommended_movies]


if __name__ == '__main__':
    # Pick a random user
    rand_user = users.sample()
    user_id = rand_user['user_id'].iloc[0]

    # Get recommendations for the random user
    recommended_movie_ids = get_movie_recommendations(user_id)

    # Get movie titles for the recommended movie IDs
    recommended_titles = movies[movies['movie_id'].isin(recommended_movie_ids)]['title'].tolist()

    print("Recommended for User ID", user_id, ":")
    print(recommended_titles)
