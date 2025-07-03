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


# Step 1: Fix the missing title
print("Before fixing missing title:")
print(messy_movies['title'])

# Fill missing title with a placeholder
messy_movies['title'] = messy_movies['title'].fillna('Unknown Movie')
print("\nAfter fixing missing title:")
print(messy_movies['title'])

# Step 2: Standardize capitalization
messy_movies['title'] = messy_movies['title'].str.title()
print("\nAfter fixing capitalization:")
print(messy_movies['title'])

# "mean imputation."
# Other strategies you'll see:

# Median imputation - use the middle value instead of average (good when you have outliers)
# Mode imputation - use the most common value (good for categories like genre)
# Forward fill - use the previous value (good for time series data)


# Step 3: Handle missing ratings
print(f"\nMissing ratings: {messy_movies['rating'].isna().sum()}")
# Fill with the average rating
avg_rating = messy_movies['rating'].mean()
messy_movies['rating'] = messy_movies['rating'].fillna(avg_rating)
print(f"Filled missing rating with average: {avg_rating:.1f}")

# Step 4: Clean the budget column
print("\nBudget column before cleaning:")
print(messy_movies['budget'])

# Convert budget from text to numbers
def clean_budget(budget_str):
    if budget_str == 'unknown':
        return None  # We'll handle this separately
    else:
        # Remove 'M' and convert to number (in millions)
        return float(budget_str.replace('M', ''))

messy_movies['budget_millions'] = messy_movies['budget'].apply(clean_budget)
print("\nAfter converting to numbers:")
print(messy_movies['budget_millions'])

# Now handle the missing budget with mean imputation
avg_budget = messy_movies['budget_millions'].mean()
messy_movies['budget_millions'] = messy_movies['budget_millions'].fillna(avg_budget)
print(f"\nFilled missing budget with average: {avg_budget:.1f}M")
print(messy_movies['budget_millions'])

# Step 5: Calculate z-scores to understand our data better
# Z-scores show how many standard deviations away from average each value is:
# 0 = exactly average, ±1 = normal, ±2 = still normal, ±3+ = outlier

print("\n--- Analyzing Budget Distribution ---")
budget_z_scores = (messy_movies['budget_millions'] - messy_movies['budget_millions'].mean()) / messy_movies['budget_millions'].std()
print("Budget z-scores:", budget_z_scores.values)

print("\n--- Analyzing Rating Distribution ---") 
rating_z_scores = (messy_movies['rating'] - messy_movies['rating'].mean()) / messy_movies['rating'].std()
print("Rating z-scores:", rating_z_scores.values)

# Show which movies are outliers
print("\n--- Final cleaned dataset ---")
print(messy_movies[['title', 'rating', 'budget_millions']])