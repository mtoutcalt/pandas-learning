import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# Let's predict if a movie will be a "blockbuster" (rating > 8.5)
movies = pd.DataFrame({
    'title': ['The Matrix', 'Inception', 'Interstellar', 'The Dark Knight', 
              'Pulp Fiction', 'Avatar', 'Titanic', 'The Avengers', 'Transformers'],
    'budget_millions': [63, 160, 165, 185, 8, 237, 200, 220, 150],
    'year': [1999, 2010, 2014, 2008, 1994, 2009, 1997, 2012, 2007],
    'rating': [8.7, 8.8, 8.6, 9.0, 8.9, 7.8, 7.9, 8.0, 6.0]
})

# Create our target: is it a blockbuster? (rating > 8.5)
movies['is_blockbuster'] = movies['rating'] > 8.5
print("Our data with blockbuster labels:")
print(movies[['title', 'budget_millions', 'year', 'rating', 'is_blockbuster']])

# Train a decision tree
# "Can we predict if a movie will be a blockbuster WITHOUT knowing its rating?"
X = movies[['budget_millions', 'year']]
y = movies['is_blockbuster']

tree_model = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_model.fit(X, y)

# Test it on a new movie
new_movie = [[120, 2021]]
prediction = tree_model.predict(new_movie)
print(f"\nWill a $120M movie from 2021 be a blockbuster? {prediction[0]}")


# Let's see HOW the decision tree makes decisions
feature_names = ['budget_millions', 'year']
class_names = ['Not Blockbuster', 'Blockbuster']

plt.figure(figsize=(12, 8))
tree.plot_tree(tree_model, 
               feature_names=feature_names,
               class_names=class_names,
               filled=True)
plt.title("How the Decision Tree Thinks")
plt.show()

# Let's also see the decision rules in text
print("\nDecision Tree Rules:")
tree_rules = tree.export_text(tree_model, feature_names=feature_names)
print(tree_rules)