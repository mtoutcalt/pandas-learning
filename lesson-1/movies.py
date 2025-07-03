import pandas as pd

# Create a simple dataset
movies = pd.DataFrame({
    'title': ['The Matrix', 'Inception', 'Interstellar', 'The Dark Knight', 'Pulp Fiction'],
    'year': [1999, 2010, 2014, 2008, 1994],
    'rating': [8.7, 8.8, 8.6, 9.0, 8.9],
    'genre': ['Sci-Fi', 'Sci-Fi', 'Sci-Fi', 'Action', 'Crime']
})

# Let's explore our data
print("Our movie dataset:")
print(movies)
print(f"\nAverage rating: {movies['rating'].mean():.1f}")
print(f"Highest rated: {movies.loc[movies['rating'].idxmax(), 'title']}")
print(f"Movies from 2000s: {len(movies[movies['year'] >= 2000])}")

# Group by genre and see average ratings
print("\nAverage rating by genre:")
genre_ratings = movies.groupby('genre')['rating'].mean()
print(genre_ratings)

# Find movies with rating above average
avg_rating = movies['rating'].mean()
above_avg = movies[movies['rating'] > avg_rating]
print(f"\nMovies above average ({avg_rating:.1f}):")
print(above_avg[['title', 'rating']])


# Add a new column based on existing data
movies['decade'] = (movies['year'] // 10) * 10
movies['is_classic'] = movies['year'] < 2000

print("\nMovies with new columns:")
print(movies[['title', 'year', 'decade', 'is_classic']])

# Count movies by decade
print("\nMovies by decade:")
print(movies['decade'].value_counts().sort_index())

# Simple rule-based "prediction"
movies['predicted_popular'] = (movies['rating'] > 8.5) & (movies['year'] > 2000)
print("\nOur 'predictions':")
print(movies[['title', 'rating', 'year', 'predicted_popular']])