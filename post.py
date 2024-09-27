import os
import pandas as pd
import numpy as np
from math import radians, cos, sin, sqrt, atan2
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')

# Function to calculate haversine distance between two lat/lon points
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in kilometers
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c  # Distance in kilometers
    return distance

# Function to calculate similarity score between user input and a post row
def calculate_similarity(user_input, post_row):
    # Check for allergic ingredients
    allergic_foods = user_input.get('allergic_food', [])
    ingredients_list = post_row['ingredients'].lower().split(', ')
    
    # If any allergic food is found in the post ingredients, return a low similarity score
    if any(allergen.lower() in ingredients_list for allergen in allergic_foods):
        return 0  # Exclude this post from recommendations

    score = 0

    # Cuisine Preference
    if user_input['cuisine'] == post_row['cuisine']:
        score += 50  # Higher priority for cuisine match

    # Veg/Non-Veg Preference
    if user_input['veg_or_nonveg'] == post_row['veg_non_veg']:
        score += 30  # Veg/Non-Veg match

    # Location Proximity - Calculate the inverse of the distance
    distance = haversine(user_input['latitude'], user_input['longitude'], post_row['latitude'], post_row['longitude'])
    score += (1 / (distance + 1)) * 100  # Inverse weight for distance

    # Taste Preferences (Sweetness, Sourness, Spiciness)
    taste_diff = abs(user_input['sweetness'] - post_row['Sweetness']) + \
                 abs(user_input['sourness'] - post_row['Sourness']) + \
                 abs(user_input['spice_level'] - post_row['Spice Level'])
    taste_similarity = max(0, 100 - taste_diff)  # Inverse similarity for taste preferences
    score += taste_similarity

    # Number of Likes (add points based on post popularity)
    score += post_row['no_of_likes']  # Add likes as a direct weight

    return score

# Load the posts dataset (assuming post.csv contains relevant data)
posts_df = pd.read_csv('post2.csv')

# Define user input including allergic food
user_input = {
    'latitude': 28.7041,  # Example latitude (Delhi)
    'longitude': 77.1025,  # Example longitude (Delhi)
    'cuisine': 'Punjabi',
    'veg_or_nonveg': 'Non-Veg',
    'sweetness': 5,  # User preference for sweetness
    'sourness': 3,   # User preference for sourness
    'spice_level': 7,  # User preference for spice level
    'followers': 1000,  # Number of followers
    'allergic_food': ['tomatoes', 'ghee']  # Example allergic foods
}

# Calculate similarity scores for all posts
posts_df['similarity_score'] = posts_df.apply(lambda row: calculate_similarity(user_input, row), axis=1)

# Sort all users based on similarity score in descending order
posts_df_sorted = posts_df[['post_id', 'similarity_score']].sort_values(by='similarity_score', ascending=False)

# Store the sorted results in a list of tuples (post_id, similarity_score)
all_users_similarities = list(posts_df_sorted.itertuples(index=False, name=None))

# Create a list for rejected posts
rejected_posts = posts_df[posts_df['similarity_score'] == 0][['post_id', 'ingredients']].reset_index(drop=True)

# Iterate over the list and print post_id and similarity score
for user in all_users_similarities:
    print(f"Post ID: {user[0]}, Similarity Score: {user[1]:.2f}")

# Print rejected posts
print("\nRejected Posts (due to allergic ingredients):")
for index, row in rejected_posts.iterrows():
    print(f"Post ID: {row['post_id']}, Ingredients: {row['ingredients']}")
