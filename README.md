<img width="1794" height="833" alt="Screenshot 2026-05-06 141345" src="https://github.com/user-attachments/assets/a3984f61-901e-4004-85c2-c61743466c15" /># 🔕 Silent Churn Detector

> Detecting behavioral drift in telecom customers before they leave.

## 📌 Problem Statement
Traditional churn models are reactive — they predict who will leave after
the signs are already obvious. This project detects **silent churn** —
the gradual behavioral disengagement that happens weeks before cancellation.

## 🚀 What Makes This Unique
Instead of basic churn prediction, we engineer 4 custom behavioral drift
features that don't exist in the raw dataset:

| Feature | Formula | Meaning |
|---|---|---|
| Engagement Score | tenure ÷ MonthlyCharges | Value vs cost ratio |
| Drift Risk Score | MonthlyCharges ÷ services_used | Usage vs cost ratio |
| Charge Pressure | MonthlyCharges ÷ TotalCharges | Financial stress signal |
| Loyalty Index | tenure × services_used | True depth of relationship |

## 📊 Dataset
IBM Telco Customer Churn Dataset from Kaggle.
- 7,043 customers
- 21 features
- 26% churn rate

Download the dataset from:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Place the CSV file in the same folder as app.py before running.

## 🛠️ Tech Stack
- Python
- Scikit-learn (Random Forest)
- Pandas, NumPy
- Matplotlib, Seaborn
- Streamlit

## ▶️ How to Run

1. Clone the repo:
git clonegit clone https://github.com/Thrisha83/silent-churn-detector.git

2. Install dependencies:
pip install -r requirements.txt

3. Download the dataset from Kaggle and place it in the folder

4. Run the dashboard:
streamlit run app.py

## 📈 Model Performance
- Accuracy: ~85%
- ROC-AUC Score: ~0.84
- Algorithm: Random Forest Classifier (100 estimators)

## 🖥️ Dashboard Features
- Overview — dataset stats and churn rate
- Drift Signals — behavioral drift visualisations
- Model Insights — feature importance and confusion matrix
- Live Predictor — real time churn risk scoring


## 📸 Dashboard Screenshots

### Overview
![Overview](C:\Users\Thrisha\OneDrive\Pictures\Screenshots 1\Screenshot 2026-05-06 103541.png)

### Drift Signals
![Drift Signals](C:\Users\Thrisha\OneDrive\Pictures\Screenshots 1\Screenshot 2026-05-06 153033.png)

### Model Insights
![Model Insights]("C:\Users\Thrisha\OneDrive\Pictures\Screenshots 1\Screenshot 2026-05-06 021315.png")

### Live Predictor — High Risk
![Live Predictor](C:\Users\Thrisha\OneDrive\Pictures\Screenshots 1\Screenshot 2026-05-06 130750.png)

## 👤 Author
Thrisha — github.com/Thrisha83
