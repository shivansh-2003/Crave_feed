import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# Function to calculate distance between two cities using GPT model with validation
def gpt_city_distance(city1, city2, openai_api_key):
    try:
        # Create a prompt template for asking the distance
        prompt_template = PromptTemplate(
            input_variables=["city1", "city2"],
            template="Estimate the distance in kilometers between {city1} and {city2} in India."
        )

        # Format the prompt with the two cities
        prompt = prompt_template.format(city1=city1, city2=city2)

        # Initialize the ChatOpenAI model with the API key and other parameters
        chat_model = ChatOpenAI(
            openai_api_key=openai_api_key,  # Your API Key
            model="gpt-3.5-turbo",  # Model name
            temperature=0,  # Optional: Controls randomness
            max_retries=3  # Optional: Retry attempts for failed requests
        )

        # Send the prompt to GPT and get the response
        response = chat_model([HumanMessage(content=prompt)])

        # Extract the distance from GPT's response
        distance = response[0].content.strip()

        # Try converting the response into a float (distance in kilometers)
        try:
            return float(distance.split()[0])  # Assuming distance is the first word
        except ValueError:
            return 1000  # Default large distance if parsing fails
    except Exception as e:
        print(f"Error in GPT city distance calculation: {e}")
        return 1000  # Default large distance in case of any error

# Function to calculate similarity score between users
def calculate_similarity(user_input, user_row, openai_api_key):
    similarity_score = 0

    # Cuisine similarity (High priority)
    if user_input['favourite_cuisine'] == user_row['favourite_cuisine']:
        similarity_score += 50

    # Veg/Non-Veg preference match (Medium priority)
    if user_input['veg_or_nonveg'] == user_row['veg_or_nonveg']:
        similarity_score += 30

    # Location similarity using GPT for estimating distance
    location_distance = gpt_city_distance(user_input['location'], user_row['location'], openai_api_key)
    location_similarity = 1 / (1 + location_distance)  # Inverse relationship with distance
    similarity_score += location_similarity * 10  # Small weight for location

    # Ensure 'no_of_followers' is a valid number
    try:
        followers_weight = int(user_row['no_of_followers']) / 5000  # Convert string to integer
    except ValueError:
        followers_weight = 0  # Handle invalid followers count gracefully
    similarity_score += followers_weight * 10  # Small weight for followers

    return similarity_score

# Function to calculate and return similarity scores for all users
def calculate_all_similarities(user_input, df, openai_api_key):
    similarity_scores = []

    # Iterate over each user in the dataset and calculate similarity score
    for _, user_row in df.iterrows():
        if user_input['user_id'] == user_row['user_id']:  # Skip the same user
            continue

        try:
            similarity_score = calculate_similarity(user_input, user_row, openai_api_key)
            similarity_scores.append((user_row['user_id'], similarity_score))
        except Exception as e:
            print(f"Error calculating similarity for user {user_row['user_id']}: {e}")

    # Sort users by similarity score in descending order
    sorted_similarities = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    return sorted_similarities

# Example: User input for recommendation
user_input = {
    'user_id': 101,  # Input user ID (can be unique)
    'location': 'Delhi',
    'favourite_cuisine': 'Gujarati',
    'veg_or_nonveg': 'Non-Veg',
    'no_of_followers': 2000
}

# Sample DataFrame (Assuming a CSV file loaded with similar columns)
df = pd.read_csv('user.csv')

# OpenAI API key as an argument
openai_api_key = "your_openai_api_key_here"

# Calculate similarities for all users
all_users_similarities = calculate_all_similarities(user_input, df, openai_api_key)

# Display users from highest to lowest similarity score
print("Recommended users from highest to lowest similarity score:")
for user in all_users_similarities:
    print(f"User ID: {user[0]}, Similarity Score: {user[1]:.2f}")
