import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

print("=== LESSON 5: EXPLORING DATA & MAKING BETTER FEATURES ===")
print("Goal: Learn how to dig deeper into data and create better predictions!\n")

# Let's start with a bigger, more realistic movie dataset
movies = pd.DataFrame({
    'title': ['The Matrix', 'Inception', 'Interstellar', 'The Dark Knight', 
              'Pulp Fiction', 'Avatar', 'Titanic', 'The Avengers', 'Transformers',
              'Toy Story', 'Shrek', 'Finding Nemo', 'The Lion King', 'Frozen'],
    'budget_millions': [63, 160, 165, 185, 8, 237, 200, 220, 150, 30, 60, 94, 45, 150],
    'year': [1999, 2010, 2014, 2008, 1994, 2009, 1997, 2012, 2007, 1995, 2001, 2003, 1994, 2013],
    'rating': [8.7, 8.8, 8.6, 9.0, 8.9, 7.8, 7.9, 8.0, 6.0, 8.3, 7.9, 8.2, 8.5, 7.4],
    'runtime_minutes': [136, 148, 169, 152, 154, 162, 194, 143, 144, 81, 90, 100, 88, 102],
    'genre': ['Sci-Fi', 'Sci-Fi', 'Sci-Fi', 'Action', 'Crime', 'Sci-Fi', 'Romance', 'Action', 'Action',
              'Animation', 'Animation', 'Animation', 'Animation', 'Animation']
})

print("Our movie dataset:")
print(movies.head())
print(f"We have {len(movies)} movies to analyze\n")

# ================================
# STEP 1: BASIC EXPLORATION
# ================================
print("STEP 1: Let's explore our data!")
print("-" * 40)

# What are the basic stats?
print("Basic statistics (this gives us the big picture):")
print(movies[['budget_millions', 'year', 'rating', 'runtime_minutes']].describe())

# What's the most common genre?
print("\nHow many movies of each genre do we have?")
print(movies['genre'].value_counts())

# Which genre has the best ratings on average?
print("\nAverage rating by genre:")
# "Take all the movies, group them by genre, find the average rating for each genre, then show me the results from best to worst."
genre_ratings = movies.groupby('genre')['rating'].mean().sort_values(ascending=False)
print(genre_ratings)

# ================================
# STEP 2: FINDING RELATIONSHIPS
# ================================
print("STEP 2: Do any features relate to each other?")
print("-" * 40)

# Correlation tells us if two things tend to go up/down together
# Values close to 1 = strong positive relationship
# Values close to -1 = strong negative relationship  
# Values close to 0 = no relationship

correlations = movies[['budget_millions', 'year', 'rating', 'runtime_minutes']].corr()
print("Correlation matrix:")
print(correlations.round(2)) # 2 decimal points

print("\n💡 Let's interpret this:")
budget_rating_corr = correlations.loc['budget_millions', 'rating'] # get the specific location - col
year_rating_corr = correlations.loc['year', 'rating']
runtime_rating_corr = correlations.loc['runtime_minutes', 'rating']

print(f"• Budget vs Rating: {budget_rating_corr:.2f} (weak relationship)")
print(f"• Year vs Rating: {year_rating_corr:.2f} (very weak relationship)")
print(f"• Runtime vs Rating: {runtime_rating_corr:.2f} (almost no relationship)")
print("• Bigger budgets don't guarantee better ratings!")

# strongest correlation is budget vs. year - newer movies have bigger budgets 
# second strongest - bigger budgets tend to be longer movies
# weakest correlation - higher budgets have lower ratings


# ================================
# STEP 3: CREATING NEW FEATURES
# ================================
print("\nSTEP 3: Let's create some new, smarter features!")
print("-" * 40)

# Start with a copy so we don't mess up the original
movies_enhanced = movies.copy()

# Feature 1: How old is each movie?
current_year = 2024
movies_enhanced['age_years'] = current_year - movies_enhanced['year']
print("Feature 1 - Movie age:")
print(movies_enhanced[['title', 'year', 'age_years']].head())

# Feature 2: What decade is it from?
movies_enhanced['decade'] = (movies_enhanced['year'] // 10) * 10 # drop singles digit then multiply by 10
print("\nFeature 2 - Decade:")
print(movies_enhanced[['title', 'year', 'decade']].head())

# Feature 3: Budget categories (Low, Medium, High)
# We'll use percentiles to split into 3 equal groups
low_budget = movies_enhanced['budget_millions'].quantile(0.33)  # Bottom 33%
high_budget = movies_enhanced['budget_millions'].quantile(0.67)  # Top 33%

def categorize_budget(budget):
    if budget <= low_budget:
        return 'Low'
    elif budget <= high_budget:
        return 'Medium'
    else:
        return 'High'

movies_enhanced['budget_category'] = movies_enhanced['budget_millions'].apply(categorize_budget)
print("\nFeature 3 - Budget categories:")
print(f"Low budget: Under ${low_budget:.0f}M")
print(f"Medium budget: ${low_budget:.0f}M - ${high_budget:.0f}M")
print(f"High budget: Over ${high_budget:.0f}M")
print(movies_enhanced[['title', 'budget_millions', 'budget_category']].head())

# Feature 4: Is it a classic? (old + high rating)
movies_enhanced['is_classic'] = (movies_enhanced['age_years'] > 20) & (movies_enhanced['rating'] > 8.0)
print("\nFeature 4 - Classic movies (old + highly rated):")
classics = movies_enhanced[movies_enhanced['is_classic']]
print(f"Found {len(classics)} classics:")
print(classics[['title', 'year', 'rating', 'age_years']])

# Feature 5: Rating efficiency (rating per dollar spent)
movies_enhanced['rating_per_budget'] = movies_enhanced['rating'] / movies_enhanced['budget_millions']
print("\nFeature 5 - Most efficient movies (best rating per dollar):")
efficient = movies_enhanced.nlargest(5, 'rating_per_budget')
print(efficient[['title', 'rating', 'budget_millions', 'rating_per_budget']])

# ================================
# STEP 4: ANALYZE OUR NEW FEATURES
# ================================
print("\nSTEP 4: What do our new features tell us?")
print("-" * 40)

# Which budget category makes the best movies?
print("Average rating by budget category:")
budget_analysis = movies_enhanced.groupby('budget_category')['rating'].mean().sort_values(ascending=False)
print(budget_analysis)

# Which decade made the best movies?
print("\nAverage rating by decade:")
decade_analysis = movies_enhanced.groupby('decade')['rating'].mean().sort_values(ascending=False)
print(decade_analysis)

print("\n💡 INSIGHTS:")
print("• Low budget movies often have higher ratings!")
print("• 1990s movies have the highest average rating")
print("• Animation movies are very efficient (high rating per dollar)")

# ================================
# STEP 5: BETTER MACHINE LEARNING
# ================================
print("\nSTEP 5: Can our new features make better predictions?")
print("-" * 40)

# First, let's prepare our data for machine learning
# We need to convert text categories to numbers
movies_ml = movies_enhanced.copy()

# Convert genre to numbers (0=Action, 1=Animation, 2=Crime, etc.)
# dictionary comprehension - first give me unique generes, then loop
# {key_expression: value_expression for item in iterable}
# Combine two lists into a dictionary
# names = ['Alice', 'Bob', 'Charlie']
# ages = [25, 30, 35]
# people = {name: age for name, age in zip(names, ages)}
# Result: {'Alice': 25, 'Bob': 30, 'Charlie': 35}

# The term "comprehension" comes from set comprehension in mathematics. In math, set comprehension is a way to define a set by describing the properties its elements must have. For example:
# {x² | x ∈ {1, 2, 3, 4, 5}}
# This mathematical notation reads as "the set of x-squared for all x in the set {1, 2, 3, 4, 5}."
# Python borrowed this concept and terminology:
# List comprehension (came first):
# pythonsquares = [x**2 for x in range(1, 6)]
# Dictionary comprehension (followed the same pattern):
# pythonsquares = {x: x**2 for x in range(1, 6)}
# The word "comprehension" essentially means "a complete understanding or inclusion of something." In programming context, it refers to a concise way to "comprehend" or generate a collection by expressing the rule for creating its elements.
# All of Python's comprehensions follow this same readable pattern:

# List comprehension: [expression for item in iterable]
# Dictionary comprehension: {key: value for item in iterable}
# Set comprehension: {expression for item in iterable}

# So "dictionary comprehension" is just the dictionary version of this general comprehension syntax pattern that Python uses for creating collections concisely.
genre_mapping = {genre: i for i, genre in enumerate(movies_ml['genre'].unique())}
movies_ml['genre_number'] = movies_ml['genre'].map(genre_mapping)
print("Genre mapping:", genre_mapping)

# Convert budget category to numbers
budget_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
movies_ml['budget_category_number'] = movies_ml['budget_category'].map(budget_mapping)
print("Budget category mapping:", budget_mapping)

# Compare OLD vs NEW feature sets
old_features = ['budget_millions', 'year']
new_features = ['budget_millions', 'year', 'runtime_minutes', 'age_years', 'genre_number', 'budget_category_number']

print(f"\nOLD features: {old_features}")
print(f"NEW features: {new_features}")

# Train models with both feature sets
X_old = movies_ml[old_features]
X_new = movies_ml[new_features]
y = movies_ml['rating']

# Train the models
model_old = LinearRegression()
model_old.fit(X_old, y)

model_new = LinearRegression()
model_new.fit(X_new, y)

# Test how well they predict
predictions_old = model_old.predict(X_old)
predictions_new = model_new.predict(X_new)

# Let's say the model learned: rating = 0.001×budget + 0.002×year - 3.5

# Calculate errors (lower is better)
rmse_old = np.sqrt(mean_squared_error(y, predictions_old))
rmse_new = np.sqrt(mean_squared_error(y, predictions_new))

print(f"\nModel Performance:")
print(f"Old features error: {rmse_old:.3f}")
print(f"New features error: {rmse_new:.3f}")
improvement = ((rmse_old - rmse_new) / rmse_old * 100)
print(f"Improvement: {improvement:.1f}% better!")

# ================================
# STEP 6: MAKE A PREDICTION
# ================================
print("\nSTEP 6: Let's predict a new movie!")
print("-" * 40)

# Imagine we're making a new movie with these specs:
new_movie_specs = {
    'budget_millions': 100,
    'year': 2024,
    'runtime_minutes': 120,
    'age_years': 0,  # Brand new movie
    'genre_number': 1,  # Animation (from our mapping above)
    'budget_category_number': 1  # Medium budget
}

new_movie_data = [list(new_movie_specs.values())] # make a list of lists - the list has 1 list
predicted_rating = model_new.predict(new_movie_data)[0] # predict needs a 2d array (list of lists)

print("New movie specs:")
print(f"• Budget: ${new_movie_specs['budget_millions']}M")
print(f"• Year: {new_movie_specs['year']}")
print(f"• Runtime: {new_movie_specs['runtime_minutes']} minutes")
print(f"• Genre: Animation")
print(f"• Budget category: Medium")

print(f"\n🎯 PREDICTED RATING: {predicted_rating:.1f}")

# ================================
# LESSON SUMMARY
# ================================
print("\n" + "="*50)
print("🎉 LESSON COMPLETE!")
print("="*50)

print("\n📚 WHAT YOU LEARNED:")
print("1. How to explore data systematically")
print("2. How to find relationships between features")
print("3. How to create new, meaningful features")
print("4. How new features can improve predictions")
print("5. How to make predictions for new scenarios")

print("\n🔑 KEY INSIGHTS:")
print("• Animation movies are very efficient (high rating per dollar)")
print("• Low budget movies often have higher ratings")
print("• Adding smart features improved our model by {:.1f}%".format(improvement))
print("• Feature engineering is often more important than complex algorithms")

print("\n🎯 NEXT STEPS:")
print("• Try creating your own features")
print("• Experiment with different data")
print("• Think about what features might predict success in other domains")

print("\n💡 REMEMBER: Good features often matter more than fancy algorithms!")