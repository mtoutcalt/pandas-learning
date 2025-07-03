import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Create a bigger dataset for training
movies = pd.DataFrame({
    'title': ['The Matrix', 'Inception', 'Interstellar', 'The Dark Knight', 
              'Pulp Fiction', 'Avatar', 'Titanic', 'The Avengers'],
    'budget_millions': [63, 160, 165, 185, 8, 237, 200, 220],
    'year': [1999, 2010, 2014, 2008, 1994, 2009, 1997, 2012],
    'rating': [8.7, 8.8, 8.6, 9.0, 8.9, 7.8, 7.9, 8.0]
})

print("Our movie dataset:")
print(movies)

# The magic: train a model to predict rating based on budget and year
X = movies[['budget_millions', 'year']]  # Features (inputs)
y = movies['rating']                     # Target (what we want to predict)

model = LinearRegression()
model.fit(X, y)

# Make a prediction for a new movie
new_movie = [[150, 2020]]  # $150M budget, made in 2020
predicted_rating = model.predict(new_movie)
print(f"\nPredicted rating for a $150M movie from 2020: {predicted_rating[0]:.1f}")



# Let's see what the model learned
print(f"\nModel learned these relationships:")
print(f"Budget coefficient: {model.coef_[0]:.4f}")
print(f"Year coefficient: {model.coef_[1]:.4f}")
print(f"Base rating (intercept): {model.intercept_:.2f}")

# This means: rating = base + (budget * budget_coef) + (year * year_coef)
print(f"\nFormula the model learned:")
print(f"Rating = {model.intercept_:.2f} + ({model.coef_[0]:.4f} * budget) + ({model.coef_[1]:.4f} * year)")

# Let's test a few different scenarios
test_movies = [
    [50, 2024],   # Low budget, recent
    [300, 1990],  # High budget, old
    [100, 2010]   # Medium budget, medium year
]

for budget, year in test_movies:
    pred = model.predict([[budget, year]])[0]
    print(f"${budget}M movie from {year}: predicted rating {pred:.1f}")


# The model thinks:

# Higher budgets = slightly lower ratings (maybe big budget movies are more commercial, less artistic?)
# Newer movies = higher ratings (maybe rating inflation over time, or better filmmaking?)


# Test the model on movies it was trained on
predictions = model.predict(X)
print("\nHow well does it predict movies it already knows?")
for i, (actual, predicted) in enumerate(zip(y, predictions)):
    title = movies.iloc[i]['title']
    print(f"{title}: Actual {actual}, Predicted {predicted:.1f}, Error {abs(actual-predicted):.1f}")


# What this tells us:
# Your model is okay but not great. In real ML, we'd want errors mostly under 0.2 for this kind of prediction.


# Calculate overall error
mse = mean_squared_error(y, predictions)
rmse = np.sqrt(mse)
print(f"\nOverall model performance:")
# So RMSE of 0.28 means "on average, our predictions are off by about 0.28 rating points."
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"Average error: {np.mean(np.abs(y - predictions)):.2f} rating points")

# RMSE: 0.28 = On average, predictions are off by about 0.28 rating points
# Average error: 0.22 = Typical mistake is about 0.22 points



# But here's the big problem: We tested the model on the same data we trained it on. That's like giving students the exact same test they studied from - of course they'll do well!
# In real ML, we need to test on "unseen" data:
# Let's do it properly - split data into training and testing
from sklearn.model_selection import train_test_split

# Split: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train on training data only
model_proper = LinearRegression()
model_proper.fit(X_train, y_train)

# Test on unseen data
test_predictions = model_proper.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

print(f"\nProper evaluation (unseen data):")
print(f"Test RMSE: {test_rmse:.2f}")
print(f"Training data size: {len(X_train)}")
print(f"Test data size: {len(X_test)}")