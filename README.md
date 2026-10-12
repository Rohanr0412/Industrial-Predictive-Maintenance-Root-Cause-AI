# Industrial-Predictive-Maintenance-Root-Cause-AI
Predictive maintenance and machine failure prediction using Python, XGBoost, LSTM, SHAP, and Streamlit.

# 🏭 Industrial Predictive Maintenance & Root-Cause AI

An end-to-end machine learning and deep learning project that predicts potential machine failures, identifies likely failure components, and supports proactive maintenance decisions through an interactive Streamlit dashboard.

## 🎯 Business Problem

Unexpected industrial machine failures can cause production downtime, maintenance costs, and operational disruption. This project explores how historical telemetry, error events, and maintenance-related information can support earlier identification of machines that may require attention.

## 🚀 Project Features

- 📊 Exploratory data analysis and time-based feature engineering
- 🤖 Failure prediction using Logistic Regression, Random Forest, and XGBoost
- 🧠 LSTM-based deep learning for sensor time-series analysis
- 🔍 Root-cause component classification using XGBoost
- 💡 Model explainability using SHAP and feature-importance analysis
- 🚨 Machine risk summaries and maintenance recommendations
- 📈 Interactive Streamlit dashboard with telemetry and machine-history views

## 🛠️ Technology Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · TensorFlow/Keras · SHAP · Streamlit · Plotly · Jupyter Notebook

## 📂 Repository Structure

- `cp2_app.py` — Streamlit application
- `Python_Ml_DL.ipynb` — analysis, feature engineering, model training, and evaluation workflow
- `component_label_encoder.pkl` — component label encoder
- `component_preprocessor.pkl` — component model preprocessing artifact
- `component_xgboost.json` — component classification model
- `failure_lstm.keras` — LSTM model
- `failure_preprocessor.pkl` — failure prediction preprocessing artifact
- `failure_xgboost.json` — XGBoost failure prediction model
- `lstm_scaler.pkl` — LSTM input scaler
- `requirements.txt` — Python dependency list
🖥️ Dashboard Preview

## 📊 Model Evaluation

The notebook records the following failure-prediction results on its test split:

| Model | ROC-AUC | PR-AUC | F1-score |
|---|---:|---:|---:|
| Logistic Regression | 0.9678 | 0.3188 | 0.3529 |
| Random Forest | 0.9974 | 0.8137 | 0.8295 |
| XGBoost | 0.9982 | 0.8681 | 0.8822 |

The notebook also evaluates an LSTM time-series model and a multiclass component-classification model. Results should be interpreted in the context of class imbalance and validated for potential data leakage before being used for real operational decisions.

## ▶️ Run the Streamlit App

1. Install Python.
2. Clone or download this repository.
3. Install the dependencies:

   `pip install -r requirements.txt`

4. Ensure the required deployment dataset and model files are available at the paths expected by the application.
5. Launch the dashboard:

   `streamlit run cp2_app.py`

## ⚠️ Important Notes

- This is a portfolio and analytical demonstration, not a production-certified industrial safety system.
- Predictions are estimates and should not replace qualified maintenance inspections.
- Confirm dataset provenance and feature/target construction before interpreting results as real-world failure probabilities.
- Dataset files are not included in this repository; the application requires its deployment dataset separately.

## 👨‍💻 Author

**Rohan Ram**

- GitHub: [@Rohanr0412](https://github.com/Rohanr0412)
- LinkedIn: [Rohan Ram](https://www.linkedin.com/in/rohan-ram-a8a59727/)
