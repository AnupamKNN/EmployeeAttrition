# <B>Ai Robotics Salary Prediction using Machine Learning</B>

[Link to Live Project](https://ai-robotics-employee-attrition-salary.onrender.com)

## 🏢  Company Overview

Ai Robotics is a technology-driven company at the forefront of digital transformation. With a strong emphasis on Enterprise Agile Planning, DevOps, AI Analytics, and Digital Security, the company integrates cutting-edge technologies to optimize business efficiency.

Apart from its core technological focus, Ai Robotics also plays a crucial role in managing critical business functions, including:

- Sourcing and Procurement – Ensuring streamlined vendor selection and material acquisition.
- Supply Chain Management – Optimizing inventory, logistics, and distribution for seamless operations.
- Business Intelligence & Decision-Making – Leveraging AI to drive data-informed strategies.

Recognizing that technology is only as effective as the people behind it, Ai Robotics continuously invests in talent retention and workforce stability. This commitment to employee satisfaction forms the foundation of the present initiative, which predicts employee salaries through machine learning to improve retention and hiring strategies.

---

## 📉 Problem Statement
Employee retention is one of the biggest challenges organizations face today. High employee turnover leads to significant operational inefficiencies, including:

- Increased recruitment and training costs.
- Loss of experienced professionals, impacting productivity.
- Disruptions in team dynamics and project continuity.

At Ai Robotics, Mr. Francis, a senior manager, noticed a rising turnover rate among employees. Many skilled professionals were resigning, and exit interviews revealed a common concern:
💰 Salary dissatisfaction and a lack of competitive compensation strategies.

Despite offering performance-based raises, the company lacked data-driven insights to determine the right salary adjustments. Without a structured approach to salary benchmarking, Ai Robotics risked further attrition, potentially affecting long-term business growth.

To address this, Mr. Francis sought the expertise of Mr. Andrew, a data scientist, to develop a machine learning model capable of predicting employee salaries based on market trends, skills, and experience.

---

## 💡 Business Solution
To combat employee attrition and ensure competitive compensation, Ai Robotics decided to integrate predictive analytics into its HR decision-making process.

✅ How does the solution work?

1️⃣ Data Collection & Processing:

- Gathering historical employee data (experience, job role, location, etc.).
- Incorporating external salary benchmarks for industry comparison.

2️⃣ Machine Learning Model Development:

- Training predictive models to forecast salaries based on various employee attributes.
- Using algorithms like Linear Regression, Decision Trees, and Random Forest.

3️⃣ Implementation & Decision-Making:

- The HR team can adjust salary structures proactively to retain talent.
- Managers receive AI-powered insights to ensure competitive pay scales.

By leveraging machine learning, Ai Robotics aims to:
✅ Optimize salary structures for enhanced employee satisfaction.
✅ Reduce turnover rates by offering data-backed compensation.
✅ Improve overall workforce stability, boosting productivity and retention.

This data-driven strategy aligns employee expectations with industry standards, strengthening the company’s position in a competitive talent market.

---

## ⚙️ Tech Stack

|                  Category                  |             Tools & Libraries            |
|--------------------------------------------|------------------------------------------|
| Language                                   | Python 3.10                              |
| ML Frameworks                              | Scikit-learn, XGBoost                    |  
| Data Processing                            | Pandas, NumPy                            |
| Visualization                              | Matplotlib, Seaborn                      |
| Deployment                                 | Flask, Render.com                        |
| Database                                   | MongoDB,                                 |
| CI/CD Pipelines                            | GitHub Actions                           |
| Containarization                           | Docker, GitHub Container Registry (GHCR) |
| MLOps Performance Metrics Tracking         | MLFlow, Dagshub                          |

---

## 🛠 Installation & Setup

1️⃣ Clone the Repository
- git clone https://github.com/AnupamKNN/EmployeeAttrition.git
- cd EmployeeAttrition

2️⃣ Create a Virtual Environment & Install Dependencies
- python3.10 -m venv venv
- source venv/bin/activate  # (Linux/macOS)
- venv\Scripts\activate  # (Windows)
- pip install -r requirements.txt

If you are using Conda Environment (Anaconda Navigator)
- conda create --name venv python=3.10 -y
- conda activate venv/
- pip install -r requirements.txt


### Explanation:
1. Creates a Conda virtual environment** named `venv` with Python 3.10.
2. Activates the environment.
3. Installs dependencies from the `requirements.txt` file.  

This makes it easy for anyone cloning your repo to set up their environment correctly! ✅

--- 

## 📊 Dataset Overview

The dataset consists of employee-related features such as:

- **jobId**: Unique ID representing each employee.
- **companyId**: Unique ID representing the company.
- **jobType**: The position or role of the employee in the company.
- **degree**: The degree obtained by the employee.
- **major**: The field of specialization of the employee.
- **industry**: The industry in which the employee is working.
- **yearsExperience**: Total number of years of experience of the employee.
- **milesFromMetropolis**: The distance (in miles) between the employee’s residence and the company.

📌 **Target Variable**: Predicted Salary: Employee's salary in $100k units (e.g., 250 represents $250,000).

---

## 🎯 Model Training & Evaluation
The models are trained using classification-based supervised learning algorithms from Scikit-Learn

### 📊 Model Performance Summary

#### Before Hyperparameter Tuning


#### 📊 Train Results

| Rank | Model              | MAE      | MSE      | RMSE     | R² Score (%) |
|------|-------------------|---------|---------|---------|-------------|
| 1️⃣  | GradientBoosting  | 15.7119 | 378.5919 | 19.4574 | 74.7265%    |
| 2️⃣  | Linear Regression | 15.8729 | 385.7838 | 19.6414 | 74.2463%    |
| 3️⃣  | XGBoost           | 16.3174 | 418.6037 | 20.4598 | 72.0554%    |
| 4️⃣  | SVR               | 16.3044 | 419.5428 | 20.4827 | 71.9927%    |
| 5️⃣  | RandomForest      | 16.4913 | 428.6273 | 20.7033 | 71.3863%    |
| 6️⃣  | KNeighbors        | 17.3134 | 478.8044 | 21.8816 | 68.0366%    |
| 7️⃣  | AdaBoost          | 20.9060 | 635.3366 | 25.2059 | 57.5870%    |
| 8️⃣  | DecisionTree      | 22.1796 | 822.3357 | 28.6764 | 45.1036%    |


#### 📊 Test Results

| Rank | Model              | MAE      | MSE      | RMSE     | R² Score (%) |
|------|-------------------|---------|---------|---------|-------------|
| 1️⃣  | GradientBoosting  | 15.7144 | 378.2952 | 19.4498 | 74.8385%    |
| 2️⃣  | Linear Regression | 15.8562 | 384.9180 | 19.6193 | 74.3980%    |
| 3️⃣  | SVR               | 16.3058 | 419.5654 | 20.4833 | 72.0935%    |
| 4️⃣  | XGBoost           | 16.3782 | 420.7569 | 20.5124 | 72.0143%    |
| 5️⃣  | RandomForest      | 16.5923 | 431.9478 | 20.7834 | 71.2699%    |
| 6️⃣  | KNeighbors        | 17.3495 | 480.6514 | 21.9238 | 68.0305%    |
| 7️⃣  | AdaBoost          | 20.8744 | 633.9303 | 25.1780 | 57.8355%    |
| 8️⃣  | DecisionTree      | 22.3757 | 825.7567 | 28.7360 | 45.0766%    |

---

### After Hyperparameter Tuning


#### 📊 Train Results

| Rank | Model               | MAE      | MSE      | RMSE     | R² Score (%) |
|------|--------------------|---------|---------|---------|-------------|
| 1️⃣  | XGBoost            | 15.5574 | 369.6906 | 19.2273 | 75.3207%    |
| 2️⃣  | GradientBoosting   | 15.5984 | 372.3011 | 19.2951 | 75.1464%    |
| 3️⃣  | SVR                | 15.6111 | 374.6397 | 19.3556 | 74.9903%    |
| 4️⃣  | Linear Regression  | 15.8675 | 385.9559 | 19.6458 | 74.2349%    |
| 5️⃣  | RandomForest       | 16.2631 | 413.8934 | 20.3444 | 72.3698%    |
| 6️⃣  | KNeighbors         | 16.6113 | 441.6813 | 21.0162 | 70.5148%    |
| 7️⃣  | DecisionTree       | 18.2529 | 535.6820 | 23.1448 | 64.2396%    |
| 8️⃣  | AdaBoost           | 20.7901 | 620.4229 | 24.9083 | 58.5826%    |

### 📊 Test Results

| Rank | Model               | MAE      | MSE      | RMSE     | R² Score (%) |
|------|--------------------|---------|---------|---------|-------------|
| 1️⃣  | XGBoost            | 15.5470 | 368.9387 | 19.2078 | 75.4608%    |
| 2️⃣  | GradientBoosting   | 15.5989 | 372.0745 | 19.2892 | 75.2523%    |
| 3️⃣  | SVR                | 15.5973 | 374.1039 | 19.3418 | 75.1173%    |
| 4️⃣  | Linear Regression  | 15.8437 | 384.8047 | 19.6164 | 74.4055%    |
| 5️⃣  | RandomForest       | 16.3258 | 416.5633 | 20.4099 | 72.2932%    |
| 6️⃣  | KNeighbors         | 16.8058 | 446.8640 | 21.1392 | 70.2778%    |
| 7️⃣  | DecisionTree       | 18.2927 | 537.8984 | 23.1926 | 64.2228%    |
| 8️⃣  | AdaBoost           | 20.7498 | 617.8025 | 24.8556 | 58.9082%    |

### Best Model

After hyperparameter tuning, **XGBRegressor** emerged as the best-performing model, achieving the highest R² score of **75.4508%** and superior predictive accuracy.

---

## 🚀 Usage
1️⃣ Input Employee Data (jobType, degree, major, etc.).
2️⃣ ML Model Predicts Salary based on historical trends.
3️⃣ Company Optimizes Compensation to improve retention and reduce hiring costs.

---

## 🔥 Results & Insights
📌 The AI model provides accurate salary predictions, enabling Ai Robotics to:
✅ Offer competitive salaries aligned with market trends.
✅ Reduce voluntary employee exits through better compensation.
✅ Optimize HR decision-making using data-driven insights.

---

## ✅ Final Deliverables

- 📁 Cleaned dataset & EDA and Model Training notebooks  
- 📦 Trained model saved in `.pkl`
- 🛠 Complete Deployable Project:
  - Data Ingestion, Data Validation, Data Transformation
  - Model Training Pipeline with Model Performance Metrics Tracking using MLflow
- 🚀 Flask app for model inference
- 🖥 Frontend interface for real-time predictions 

---

🌟 Star this repo if you found it helpful! 🚀
