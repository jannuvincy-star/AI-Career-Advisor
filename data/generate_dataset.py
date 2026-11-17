import random
import pandas as pd

random.seed(42)

skills = [
    "python",
    "java",
    "sql",
    "html",
    "css",
    "javascript",
    "ml",
    "statistics",
    "communication",
]

# Core skills for each career
career_profiles = {
    "AI/ML Engineer": ["python", "ml", "statistics"],
    "Data Scientist": ["python", "ml", "statistics", "sql"],
    "Data Analyst": ["python", "sql", "statistics"],
    "Software Developer": ["java", "python", "sql", "javascript"],
    "Backend Developer": ["python", "java", "sql"],
    "Frontend Developer": ["html", "css", "javascript"],
    "Full Stack Developer": ["html", "css", "javascript", "sql"],
    "Java Developer": ["java", "sql"],
    "Python Developer": ["python", "sql"],
}

rows = []

for career, core_skills in career_profiles.items():

    for _ in range(100):
        row = {}

        # Start with no skills
        for skill in skills:
            row[skill] = 0

        # Core skills are almost always present
        for skill in core_skills:
            row[skill] = 1 if random.random() < 0.95 else 0

        # Add some optional skills as noise
        for skill in skills:
            if skill not in core_skills:
                if random.random() < 0.15:
                    row[skill] = 1

        # Communication is common across careers
        row["communication"] = 1 if random.random() < 0.80 else 0

        row["career"] = career
        rows.append(row)

# Shuffle
random.shuffle(rows)

# Save
df = pd.DataFrame(rows)
df.to_csv("data/career_data.csv", index=False)

print("Dataset created successfully!")
print(f"Total records: {len(df)}")
print(f"Total careers: {df['career'].nunique()}")

print("\nCareer distribution:")
print(df["career"].value_counts())