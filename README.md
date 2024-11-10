# Movie Recommender System

This Python project implements a movie recommender system using the **MovieLens 100K dataset**. The system generates personalized movie recommendations by analyzing user ratings and similarities. Two approaches are implemented:

1. **Movie Recommender 2.0**: Uses user similarity to recommend the top 10 movies a user has not yet watched.
2. **Movie Recommender 1.5**: Uses clustering to recommend movies based on similar users.

Additionally, extra credit functionality allows for modifying recommendations by considering the genres of movies a user has watched.

## Dataset

The **MovieLens 100K dataset** contains 100,000 ratings (1-5) from 943 users on 1,682 movies. The data includes user demographics like age, gender, and occupation, as well as movie information and ratings.

## Features

- **Movie Recommender 2.0**: 
  - Computes user similarity based on ratings.
  - Recommends top 10 movies the user hasn't watched.
  - Extra Credit: Recommends movies by considering the user's genre preferences.

- **Movie Recommender 1.5**: 
  - Uses clustering to group similar users.
  - Recommends movies based on the clusters of similar users.

## Requirements

- **Python 3.x**
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib` (for data visualization, optional
