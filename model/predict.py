import pickle
import pandas as pd

# Load trained model
with open("model/career_model.pkl", "rb") as file:
    model = pickle.load(file)

# Available skills
skills = [
    "python",
    "java",
    "sql",
    "html",
    "css",
    "javascript",
    "ml",
    "statistics",
    "communication"
]

print("\n=== AI Career Advisor ===")
print("Available skills:")

for i, skill in enumerate(skills, start=1):
    print(f"{i}. {skill}")

# Get user input
choice = input("\nEnter skill numbers separated by spaces: ")

selected_numbers = choice.split()

# Create input with all skills = 0
user_data = {skill: 0 for skill in skills}

# Set selected skills to 1
for number in selected_numbers:
    index = int(number) - 1

    if 0 <= index < len(skills):
        user_data[skills[index]] = 1

# Convert to DataFrame
user = pd.DataFrame([user_data])

# Predict career
prediction = model.predict(user)

print("\nRecommended Career:", prediction[0])