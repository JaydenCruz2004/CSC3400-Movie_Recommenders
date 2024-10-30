## CSC 3400 Artifical Intelligence
## Assignment 3: Movie Recommenders

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load each data set (users, movies, and ratings).
users_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('u.user', sep='|', names= users_cols, encoding= 'latin-1')# read user data from u.user use | as the separator, names=user_cols, and 'latin-1' as the encoding

ratings_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv( 'u.data', sep= '\t', names=ratings_cols, encoding='latin-1')# Ratings data from u.data, use '\t' as separator, names=ratings_cols, encoding='latin-1')

#prints out first user id
#print(users['user_id'].iloc[0])

# The movies file contains a binary feature for each genre.
genre_cols = [
    "genre_unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]
movies_cols = ['movie_id', 'title', 'release_date', "video_release_date", "imdb_url"] + genre_cols
movies =pd.read_csv('u.item',sep='|',names=movies_cols, encoding='latin-1') # read movies information from u.item. Look at the file to figure out the appropriate separator. Use names=movie_cols, and encoding as 'latin-1'

# Since the ids in the dataset start at 1, we shift them to start at 0.
users["user_id"] = users["user_id"].apply(lambda x: str(x-1))
movies["movie_id"] = movies["movie_id"].apply(lambda x: str(x-1))
movies["year"] = movies['release_date'].apply(lambda x: str(x).split('-')[-1])
ratings["movie_id"] = ratings["movie_id"].apply(lambda x: str(x-1))
ratings["user_id"] = ratings["user_id"].apply(lambda x: str(x-1))
ratings["rating"] = ratings["rating"].apply(lambda x: float(x))


#function gets each movie and rating a user has watched
def getUserMovieProfile(user):
    #finds user's row of data given their id as a parameter
    userInfo = ratings[ratings["user_id"] == user]
    usersMovies = np.array(userInfo.loc[:,['movie_id','rating']])
    return usersMovies

#function computes  the cosine simiilarity between the two users
def compute_cosine_similarity(user1,user2):

    # array of userid, movies user watched, and rating
    user1 = getUserMovieProfile(user1)
    user2= getUserMovieProfile(user2)

    #creates local common movies between users with helper function
    common = getCommonMovies(user1,user2)

    user1rating = []
    user2rating = []

    #loop finds the rating of user1 and user2 give the common movies
    for movie in getCommonMovies(user1,user2):
        for rating in user1:
            if movie == rating[0]:
                user1rating.append(rating[1])

    for movie in getCommonMovies(user1,user2):
        for rating in user2:
            if movie == rating[0]:
                user2rating.append(rating[1])

    #changes array in order to compute cosine similarity
    user1rating = np.array([user1rating]).reshape(1, -1)
    user2rating = np.array([user2rating]).reshape(1, -1)

    #returns each users ratings and their common movies
    print(user1rating)
    print(user2rating)
    print('Common Movies')
    #print(getCommonMovies(user1,user2))

    #makes sure users have common movies before computing cosine similarity
    if len(common) == 0:
        return 0

    print("users cose similariry is: ")
    if len(user1rating) != 0 or len(user2rating) != 0:
        cos_sim = cosine_similarity(user1rating, user2rating)
    else:
        raise ValueError("no common movies were found for these two users")

    return cos_sim


#function gets the  reccomendation for entire data set
def get_recommendations(input_user):

    #arrays holding all similarity scores and the overall score which is the prob a user will like a movie
    overall_scores = {}
    similarity_scores = {}


    #finds cosine similairy for every user in the dataset
    for user in [x for x in users['user_id'] if x != input_user]:
        print(user)
        similarity_score = compute_cosine_similarity(user, input_user)

        if similarity_score <= 0:
            continue

        #creates list of movies input user has not watched
        filtered_list = []

        for x in getUserMovieProfile(user):
            if x[0] not in getCommonMovies(user,input_user):
                filtered_list.append(x)

        #calculates probability of whether user will like that unwatched movie
        for item in filtered_list:
            overall_scores.update({item[0]: item[1] * similarity_score})
            similarity_scores.update({item[0]: similarity_score})

    #checks to see if input user even has any matches to reccomend
    if len(overall_scores) == 0:
        return ['No recommendations possible']

    # Generate movie ranks by normalizing the scores
    movie_scores = np.array([[score / similarity_scores[item], item]
                             for item, score in overall_scores.items()], dtype=object)

    # Sort in decreasing order
    movie_scores = movie_scores[np.argsort(movie_scores[:, 0])[::-1]]

    # Extract the movie recommendations as movie ids
    movie_recommendations = [movie for _, movie in movie_scores]


    getMovieRecTitle(movie_recommendations)
    return movie_recommendations


#finds common movies between two users
def getCommonMovies(user1, user2):
    common_movies = []

    #parses through users profile to see matches
    for item in user1:
        for item2 in user2:
            if item[0] == item2[0]:
                common_movies.append(item[0])
    #print(common_movies)
    return common_movies

#finds movie title given an array of movie ids
def getMovieRecTitle(movieRec):

    #holds movie title
    movieTitleRec = []

    #looks through movie data set but only the columns of movie id and title
    movieList = np.array(movies.loc[:,['movie_id','title']])

    #loops through first 10 movieids to search for titles
    for movieid in movieRec[0:10]:
        for x in movieList:
            if x[0] == movieid:
                #updates matching movie-ids to movie rec lists
                movieTitleRec.append(x[1])

    #prints out top 10 recommended
    print("The top 10 reccomendations are we have for you are: ")
    print(movieTitleRec)

if __name__ == '__main__':
    #pick a random row from u.user file  to compare from each other
    rand_user = users.sample()

    #keeps user id of random user
    user_id = rand_user['user_id'].iloc[0]

    #list rand_user's list of movies
    rand_user_movies = ratings.loc[ratings['user_id'] == user_id]

    #makes an array for random movies
    rand_user_movieids = np.array(rand_user_movies.loc[:,['movie_id','rating']])

    get_recommendations(user_id)
