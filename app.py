from flask import Flask, render_template, request, jsonify, send_file
import pickle
import os, sys
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
from employeeattrition.exception.exception import EmployeeAttritionException

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Paths to model and preprocessor
MODEL_PATH = "final_model/model.pkl"
PREPROCESSOR_PATH = "final_model/preprocessor.pkl"

# Load model and preprocessor
model = pickle.load(open(MODEL_PATH, "rb")) if os.path.exists(MODEL_PATH) else None
preprocessor = pickle.load(open(PREPROCESSOR_PATH, "rb")) if os.path.exists(PREPROCESSOR_PATH) else None

# Dropdown options
OPTIONS = {
    "job_types": ["Enter job title", "CFO", "CEO", "VICE_PRESIDENT", "MANAGER", "JUNIOR", "JANITOR", "CTO", "SENIOR"],
    "degrees": ["Enter degree", "MASTERS", "HIGH_SCHOOL", "DOCTORAL", "BACHELORS", "NONE"],
    "majors": ["Enter major", "MATH", "NONE", "PHYSICS", "CHEMISTRY", "COMPSCI", "BIOLOGY", "LITERATURE", "BUSINESS", "ENGINEERING"],
    "industries": ["Enter industry","HEALTH", "WEB", "AUTO", "FINANCE", "EDUCATION", "OIL", "SERVICE"]
}

@app.route('/')
def home():
    return render_template('index.html', **OPTIONS)

@app.route("/about")  # About Page

def about():
    return render_template("about.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html', **OPTIONS, predicted_salary=None)
    
    try:
        data = request.form if request.content_type == 'application/x-www-form-urlencoded' else request.json

        job_type = data.get("jobType", "").strip()
        degree = data.get("degree", "").strip()
        major = data.get("major", "").strip()
        industry = data.get("industry", "").strip()
        years_experience = data.get("yearsExperience", "").strip()
        miles_from_metropolis = data.get("milesFromMetropolis", "").strip()

        if not all([job_type, degree, major, industry, years_experience, miles_from_metropolis]):
            return render_template('predict.html', **OPTIONS, predicted_salary="All fields are required.")

        try:
            years_experience = float(years_experience)
            miles_from_metropolis = float(miles_from_metropolis)
        except ValueError:
            return render_template('predict.html', **OPTIONS, predicted_salary="Invalid numerical input.")

        job_type_idx = OPTIONS["job_types"].index(job_type)
        degree_idx = OPTIONS["degrees"].index(degree)
        major_idx = OPTIONS["majors"].index(major)
        industry_idx = OPTIONS["industries"].index(industry)

        features_df = pd.DataFrame([[job_type_idx, degree_idx, major_idx, industry_idx, years_experience, miles_from_metropolis]],
                                   columns=["jobType", "degree", "major", "industry", "yearsExperience", "milesFromMetropolis"])

        if preprocessor:
            features = preprocessor.transform(features_df)
        else:
            return render_template('predict.html', **OPTIONS, predicted_salary="Preprocessor not available.")

        if model:
            predicted_salary = model.predict(features)[0] * 1000
            formatted_salary = f"${predicted_salary:,.2f}"
            return render_template('predict.html', **OPTIONS, predicted_salary=formatted_salary)
        else:
            return render_template('predict.html', **OPTIONS, predicted_salary="Model not available.")

    except Exception as e:
        return render_template('predict.html', **OPTIONS, predicted_salary=f"Error: {str(e)}")

@app.route('/batch_predict', methods=['GET', 'POST'])
def batch_predict():
    if request.method == 'GET':
        return render_template('batch_predict.html')
    
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400
    
    filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(filepath)
    
    try:
        df = pd.read_csv(filepath)
        df.drop(columns=['jobId', 'companyId'], errors='ignore', inplace=True)

        if preprocessor:
            features = preprocessor.transform(df)
        else:
            return "Preprocessor not available", 500

        if model:
            df['Predicted_Salary'] = model.predict(features) * 1000
        else:
            return "Model not available", 500

        output_path = os.path.join(UPLOAD_FOLDER, "predictions.csv")
        df.to_csv(output_path, index=False)
        return send_file(output_path, as_attachment=True)
    
    except Exception as e:
        raise EmployeeAttritionException(e, sys)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
