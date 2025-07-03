import pandas as pd
import numpy as np

# Create some realistic messy data
messy_movies = pd.DataFrame({
    'title': ['The Matrix', 'INCEPTION', 'interstellar', 'The Dark Knight', None],
    'year': [1999, 2010, 2014, 2008, 1995],
    'rating': [8.7, 8.8, None, 9.0, 7.2],
    'budget': ['63M', '160M', '165M', '185M', 'unknown'],
    'director': ['Wachowskis', 'Nolan', 'Nolan', 'Nolan', 'Wachowskis']
})

print("Our messy data:")
print(messy_movies)
print(f"\nInfo about the data:")
print(messy_movies.info())