from pathlib import Path
import pickle

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "career_model.pkl"
FRONTEND_PATH = BASE_DIR / "frontend" / "index.html"


# ============================================================
# LOAD TRAINED ML MODEL
# ============================================================

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="AI Career Advisor",
    description="AI-powered career recommendation system",
    version="1.0.0",
)


# ============================================================
# USER SKILLS INPUT MODEL
# ============================================================

class UserSkills(BaseModel):
    python: int = 0
    java: int = 0
    sql: int = 0
    html: int = 0
    css: int = 0
    javascript: int = 0
    ml: int = 0
    statistics: int = 0
    communication: int = 0


# ============================================================
# REQUIRED SKILLS FOR EACH CAREER
# ============================================================

CAREER_SKILLS = {
    "AI/ML Engineer": [
        "python",
        "ml",
        "statistics",
    ],

    "Data Scientist": [
        "python",
        "sql",
        "ml",
        "statistics",
    ],

    "Data Analyst": [
        "python",
        "sql",
        "statistics",
    ],

    "Software Developer": [
        "python",
        "java",
        "sql",
        "javascript",
    ],

    "Backend Developer": [
        "python",
        "java",
        "sql",
    ],

    "Frontend Developer": [
        "html",
        "css",
        "javascript",
    ],

    "Full Stack Developer": [
        "html",
        "css",
        "javascript",
        "sql",
    ],

    "Java Developer": [
        "java",
        "sql",
    ],

    "Python Developer": [
        "python",
        "sql",
    ],
}


# ============================================================
# LEARNING ROADMAPS
# ============================================================

CAREER_ROADMAPS = {

    "AI/ML Engineer": [
        "Python Fundamentals",
        "NumPy & Pandas",
        "Machine Learning",
        "Statistics",
        "Model Deployment",
        "Build an AI/ML Project",
    ],

    "Data Scientist": [
        "Python",
        "SQL",
        "Statistics",
        "Pandas & NumPy",
        "Machine Learning",
        "Data Science Project",
    ],

    "Data Analyst": [
        "SQL",
        "Excel",
        "Python",
        "Pandas",
        "Statistics",
        "Data Visualization",
    ],

    "Software Developer": [
        "Programming Fundamentals",
        "Java/Python",
        "OOP",
        "DSA",
        "SQL",
        "Software Project",
    ],

    "Backend Developer": [
        "Python/Java",
        "OOP",
        "SQL",
        "REST APIs",
        "FastAPI/Spring Boot",
        "Backend Project",
    ],

    "Frontend Developer": [
        "HTML",
        "CSS",
        "JavaScript",
        "Responsive Design",
        "React",
        "Frontend Project",
    ],

    "Full Stack Developer": [
        "HTML & CSS",
        "JavaScript",
        "React",
        "Backend Development",
        "SQL",
        "Full Stack Project",
    ],

    "Java Developer": [
        "Core Java",
        "OOP",
        "Collections",
        "Exception Handling",
        "SQL",
        "Java Project",
    ],

    "Python Developer": [
        "Python Fundamentals",
        "OOP",
        "Advanced Python",
        "SQL",
        "REST APIs",
        "Python Project",
    ],
}


# ============================================================
# PROJECT SUGGESTIONS
# ============================================================

CAREER_PROJECTS = {

    "AI/ML Engineer": [
        "Sign Language Recognition using Machine Learning",
        "House Price Prediction",
        "Recommendation System",
    ],

    "Data Scientist": [
        "Customer Churn Prediction",
        "Sales Forecasting",
        "Customer Segmentation",
    ],

    "Data Analyst": [
        "Sales Dashboard",
        "Retail Customer Analysis",
        "Student Performance Analysis",
    ],

    "Software Developer": [
        "Online Examination System",
        "Task Management Application",
        "Library Management System",
    ],

    "Backend Developer": [
        "REST API for E-Commerce",
        "Student Management API",
        "Authentication & Authorization API",
    ],

    "Frontend Developer": [
        "Portfolio Website",
        "E-Commerce UI",
        "Interactive Dashboard",
    ],

    "Full Stack Developer": [
        "E-Commerce Application",
        "College Management System",
        "Job Portal",
    ],

    "Java Developer": [
        "Banking Management System",
        "Inventory Management System",
        "Online Quiz Application",
    ],

    "Python Developer": [
        "Automation Tool",
        "REST API using FastAPI",
        "Web Scraping Dashboard",
    ],
}


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/")
def home():
    return FileResponse(FRONTEND_PATH)


# ============================================================
# CAREER PREDICTION
# ============================================================

@app.post("/predict")
def predict_career(skills: UserSkills):

    # --------------------------------------------------------
    # Convert input into DataFrame
    # --------------------------------------------------------

    user_data = pd.DataFrame([
        skills.model_dump()
    ])


    # --------------------------------------------------------
    # Get prediction probabilities
    # --------------------------------------------------------

    probabilities = model.predict_proba(user_data)[0]
    careers = model.classes_


    # --------------------------------------------------------
    # Sort careers by probability
    # --------------------------------------------------------

    results = sorted(
        zip(careers, probabilities),
        key=lambda x: x[1],
        reverse=True,
    )


    # --------------------------------------------------------
    # Top 3 recommendations
    # --------------------------------------------------------

    top_three = results[:3]

    recommendations = [
        {
            "career": career,
            "confidence": round(
                float(probability) * 100,
                2,
            ),
        }
        for career, probability in top_three
    ]


    # --------------------------------------------------------
    # Best career
    # --------------------------------------------------------

    best_career = recommendations[0]["career"]


    # --------------------------------------------------------
    # User selected skills
    # --------------------------------------------------------

    user_skills = skills.model_dump()

    selected_skills = [
        skill
        for skill, value in user_skills.items()
        if value == 1
    ]


    # --------------------------------------------------------
    # Required skills for recommended career
    # --------------------------------------------------------

    required_skills = CAREER_SKILLS.get(
        best_career,
        [],
    )


    # --------------------------------------------------------
    # Missing skills
    # --------------------------------------------------------

    missing_skills = [
        skill
        for skill in required_skills
        if user_skills.get(skill, 0) == 0
    ]


    # --------------------------------------------------------
    # Matched skills
    # --------------------------------------------------------

    matched_skills = [
        skill
        for skill in required_skills
        if user_skills.get(skill, 0) == 1
    ]


    # --------------------------------------------------------
    # Career readiness score
    #
    # This is based only on the tracked required skills.
    # --------------------------------------------------------

    if required_skills:
        readiness_score = round(
            len(matched_skills)
            / len(required_skills)
            * 100,
            2,
        )
    else:
        readiness_score = 0


    # --------------------------------------------------------
    # Learning roadmap
    # --------------------------------------------------------

    roadmap = CAREER_ROADMAPS.get(
        best_career,
        [],
    )


    # --------------------------------------------------------
    # Project suggestions
    # --------------------------------------------------------

    projects = CAREER_PROJECTS.get(
        best_career,
        [],
    )


    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    return {

        "recommendations": recommendations,

        "skill_analysis": {
            "career": best_career,
            "your_skills": selected_skills,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "readiness_score": readiness_score,
        },

        "learning_roadmap": roadmap,

        "project_suggestions": projects,
    }