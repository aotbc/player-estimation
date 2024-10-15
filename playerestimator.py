import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Streamlit title and description
st.title("Football Player Ability Estimator")
st.write("Upload your match data CSV to estimate player abilities based on match outcomes.")

# File uploader widget
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

# Load data from CSV
data = pd.read_csv(uploaded_file)

# Extract player names from the columns (skip the first index column)
player_names = data.columns[1:]

# Convert player data to a NumPy array (exclude the first index column)
player_data = data.drop(columns=[data.columns[0]]).to_numpy()

# Manually set ratings for specific players (example)
# Key: player name, Value: manual rating
manual_ratings = {}

# Create a mask to identify players who have manual ratings
manual_ratings_mask = np.array([name in manual_ratings for name in player_names])

# Fill in the manual ratings in an array, leaving the others as None (to be solved by the algorithm)
initial_ratings = np.array([manual_ratings.get(name, 0) for name in player_names])

# Define the objective function, which takes into account manual ratings
def objective(player_abilities):
    total_loss = 0
    for match in player_data:
        winning_team = match == 1  # players on winning team
        losing_team = match == -1  # players on losing team
        
        # Calculate the sum of abilities for the winning and losing teams
        winning_strength = np.sum(player_abilities[winning_team])
        losing_strength = np.sum(player_abilities[losing_team])
        
        # Penalize if the losing team has a higher ability sum than the winning team
        total_loss += max(0, 1 + losing_strength - winning_strength) ** 2  # Add a margin of 1 for winning
    
    # Regularization term (L2 regularization)
    regularization = 0.1 * np.sum(player_abilities ** 2)
    
    return total_loss + regularization

# Define a new function to optimize only the abilities that are not manually set
def constrained_objective(optimized_ratings):
    # Combine manually set and optimized ratings
    full_ratings = np.copy(initial_ratings)
    full_ratings[~manual_ratings_mask] = optimized_ratings  # Only update players without manual ratings
    
    return objective(full_ratings)

# Extract the initial guess for only the players without manual ratings
initial_guess = initial_ratings[~manual_ratings_mask]

# Optimize only the abilities that aren't manually set
result = minimize(constrained_objective, initial_guess, method='BFGS')

# Combine the manually set and optimized ratings
final_ratings = np.copy(initial_ratings)
final_ratings[~manual_ratings_mask] = result.x

# Create a DataFrame with player names and abilities
player_abilities_df = pd.DataFrame({
    'Player': player_names,
    'Ability': final_ratings
})

# Sort players by ability in descending order
player_abilities_df = player_abilities_df.sort_values(by='Ability', ascending=False)

# Display players by ability
print(player_abilities_df)

# Display the player abilities in a table
st.write("Estimated Player Abilities:")
st.dataframe(player_abilities_df)
