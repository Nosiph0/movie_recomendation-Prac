#!/usr/bin/env python3

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('movies.csv', quotechar='"', skipinitialspace=True, delimiter=',')
ratings = pd.read_csv('ratings.csv', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

print("Movies Data:")
print(movies.head())
print("\nRatings Data:")
print(ratings.head())

movies['item_id'] = movies['item_id'].apply(lambda x: int(str(x).split(',')[0]))

movies['item_id'] = pd.to_numeric(movies['item_id'], errors='coerce')
ratings['item_id'] = pd.to_numeric(ratings['item_id'], errors='coerce')

movies = movies.dropna(subset=['item_id'])
ratings = ratings.dropna(subset=['item_id'])

movies['item_id'] = movies['item_id'].astype(int)
ratings['item_id'] = ratings['item_id'].astype(int)

print("\nCleaned Movies Data:")
print(movies.head())
print("\nCleaned Ratings Data:")
print(ratings.head())

def merge_data(movies, ratings):
    """Merge movies and ratings data into a single DataFrame."""
    merged_data = pd.merge(ratings, movies, on='item_id')
    print("Merged Data:")
    print(merged_data.head())
    return merged_data

def create_pivot_table(data):
    pivot_table = data.pivot_table(index='user_id', columns='title', values='rating')
    print("Pivot Table:")
    print(pivot_table.head())
    return pivot_table.fillna(0)

def calculate_similarity(pivot_table):
    if pivot_table.empty:
        return pd.DataFrame()
    similarity_matrix = cosine_similarity(pivot_table.T)
    return pd.DataFrame(similarity_matrix, index=pivot_table.columns, columns=pivot_table.columns)

def get_recommendations(movie_title, similarity_df, num_recommendations=5):
    if movie_title not in similarity_df.columns:
        print(f"Movie '{movie_title}' not found in the dataset.")
        return []

    similarity_scores = similarity_df[movie_title]
    similar_movies = similarity_scores.sort_values(ascending=False).drop(movie_title)
    return similar_movies.head(num_recommendations).index.tolist()

def main():
    data = merge_data(movies, ratings)
    if data.empty:
        print("Merged data is empty. Exiting...")
        return

    pivot_table = create_pivot_table(data)
    if pivot_table.empty:
        print("Pivot table is empty. Exiting...")
        return

    similarity_df = calculate_similarity(pivot_table)
    if similarity_df.empty:
        print("Similarity DataFrame is empty. Exiting...")
        return

    movie_title = input("Enter movie name and year in brackets: ")
    recommendations = get_recommendations(movie_title, similarity_df)
    print("Recommendations for the Movie:")
    for x, title in enumerate(recommendations, start=1):
        print(f"{x}. {title}")

if __name__ == "__main__":
    main()
