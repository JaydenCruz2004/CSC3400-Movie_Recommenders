## CSC 3400 Artifical Intelligence
## Assignment 3: Movie Recommenders.

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Import data files using the MovieLens dataset format
users_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('u.user', sep='|', names=users_cols, encoding='latin-1')

ratings_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('u.data', sep='\t', names=ratings_cols, encoding='latin-1')

# Define movie features including binary genre indicators (1 = belongs to genre, 0 = does not)
genre_cols = [
    "genre_unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]
movies_cols = ['movie_id', 'title', 'release_date', "video_release_date", "imdb_url"] + genre_cols
movies = pd.read_csv('u.item', sep='|', names=movies_cols, encoding='latin-1')

# Normalize IDs to start from 0 instead of 1 for array indexing compatibility
users["user_id"] = users["user_id"].apply(lambda x: str(x - 1))
movies["movie_id"] = movies["movie_id"].apply(lambda x: str(x - 1))
movies["year"] = movies['release_date'].apply(lambda x: str(x).split('-')[-1])
ratings["movie_id"] = ratings["movie_id"].apply(lambda x: str(x - 1))
ratings["user_id"] = ratings["user_id"].apply(lambda x: str(x - 1))
ratings["rating"] = ratings["rating"].apply(lambda x: float(x))


def getUserMovieProfile(user):

    userInfo = ratings[ratings["user_id"] == user]
    usersMovies = np.array(userInfo.loc[:, ['movie_id', 'rating']])
    return usersMovies


def compute_cosine_similarity(user1, user2):

    user1_profile = getUserMovieProfile(user1)
    user2_profile = getUserMovieProfile(user2)

    common_movies = getCommonMovies(user1_profile, user2_profile)

    # Extract ratings for movies both users have watched
    user1_ratings = []
    user2_ratings = []

    for movie in common_movies:
        for rating in user1_profile:
            if movie == rating[0]:
                user1_ratings.append(rating[1])

        for rating in user2_profile:
            if movie == rating[0]:
                user2_ratings.append(rating[1])

    # Reshape arrays for sklearn's cosine_similarity function
    user1_ratings = np.array([user1_ratings]).reshape(1, -1)
    user2_ratings = np.array([user2_ratings]).reshape(1, -1)

    print(f"User 1 ratings: {user1_ratings}")
    print(f"User 2 ratings: {user2_ratings}")
    print('Common Movies')

    # Return 0 if users have no movies in common
    if len(common_movies) == 0:
        return 0

    print("Users cosine similarity is: ")
    if len(user1_ratings) != 0 or len(user2_ratings) != 0:
        cos_sim = cosine_similarity(user1_ratings, user2_ratings)
    else:
        raise ValueError("No common movies were found for these two users")

    return cos_sim


def get_recommendations(input_user):

    overall_scores = {}  # Stores predicted ratings for unwatched movies
    similarity_scores = {}  # Stores similarity scores with other users

    # Compare input user with all other users
    for user in [x for x in users['user_id'] if x != input_user]:
        print(user)
        similarity_score = compute_cosine_similarity(user, input_user)

        # Skip users with negative or zero similarity
        if similarity_score <= 0:
            continue

        # Find movies that input user hasn't watched
        filtered_list = []
        for x in getUserMovieProfile(user):
            if x[0] not in getCommonMovies(user, input_user):
                filtered_list.append(x)

        # Calculate predicted ratings using weighted scoring
        for item in filtered_list:
            overall_scores.update({item[0]: item[1] * similarity_score})
            similarity_scores.update({item[0]: similarity_score})

    if len(overall_scores) == 0:
        return ['No recommendations possible']

    # Normalize scores and sort recommendations
    movie_scores = np.array([[score / similarity_scores[item], item]
                             for item, score in overall_scores.items()], dtype=object)
    movie_scores = movie_scores[np.argsort(movie_scores[:, 0])[::-1]]

    movie_recommendations = [movie for _, movie in movie_scores]
    getMovieRecTitle(movie_recommendations)
    return movie_recommendations


def getCommonMovies(user1, user2):

    common_movies = []
    for item in user1:
        for item2 in user2:
            if item[0] == item2[0]:
                common_movies.append(item[0])
    return common_movies


def getMovieRecTitle(movieRec):

    movieTitleRec = []
    movieList = np.array(movies.loc[:, ['movie_id', 'title']])

    # Get titles for top 10 recommendations
    for movieid in movieRec[0:10]:
        for x in movieList:
            if x[0] == movieid:
                movieTitleRec.append(x[1])

    print("\nTop 10 Movie Recommendations:")
    for i, title in enumerate(movieTitleRec, 1):
        print(f"{i}. {title}")


if __name__ == '__main__':
    # Generate recommendations for a random user
    rand_user = users.sample()
    user_id = rand_user['user_id'].iloc[0]
    rand_user_movies = ratings.loc[ratings['user_id'] == user_id]
    rand_user_movieids = np.array(rand_user_movies.loc[:, ['movie_id', 'rating']])

    print(f"Generating recommendations for user {user_id}")
    get_recommendations(user_id)