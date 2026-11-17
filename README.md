# 🤖 AI Career Advisor

An AI-powered career recommendation system that suggests suitable career paths based on users' technical skills.

## Features
- Top 3 Career Recommendations
- Machine Learning-based Prediction
- Skill Gap Analysis
- Career Readiness Score
- Personalized Learning Roadmap
- Recommended Projects

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest Classifier
- FastAPI
- HTML, CSS, JavaScript

## Model Performance
- Dataset: 900 synthetic records
- Career Categories: 9
- Test Accuracy: 73.89%

Note: Results are prototype evaluation results using synthetic data.

## How to Run

1. Clone the repository.
2. Install dependencies:

   pip install -r requirements.txt

3. Start the application:

   uvicorn backend.main:app --reload

4. Open in browser:

   http://127.0.0.1:8000

## Project Structure

- backend/ - FastAPI application
- data/ - Dataset and generator
- frontend/ - Web interface
- model/ - Training and prediction scripts

## Author
Jannu Vincy
