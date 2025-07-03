# 🌾 Farmer Support Prediction App

![Farmer](farmer.png)

This project is a **Streamlit-powered machine learning web app** designed to help identify farmers who are most likely to benefit from agricultural support programs -- specifically an agro-credit. By leveraging user-provided features such as farm size, access to irrigation, income, and years of experience, the app predicts the likelihood that a farmer qualifies for support. This app is a go-to for investors who aim to rely on data rather than guesswork in choosing beneficiaries.

---

## 🚀 Overview

Many agricultural development programs face challenges in targeting the right beneficiaries. This app uses a trained **XGBoost model** to address that problem by predicting which farmers should receive support based on key socioeconomic and farm-related variables.

---

## 🧠 Features

- 🎯 **ML Model:** XGBoost Classifier  
- 🧮 **Input Variables:**
  - Gender
  - Age
  - Education Level
  - Household Size
  - Farm Size (acres)
  - Crop Type
  - Access to Irrigation
  - Total Annual Income
  - Years of Farming Experience
  - ...plus engineered features:
    - `farm_size_irrigation`
    - `income_experience`
- 📊 Predicts: **Eligibility for Agricultural Support**

---

## 📦 Files in This Repo

- `app.py` – Streamlit app source code  
- `model.pkl` – Trained XGBoost model  
- `scaler.pkl` – Preprocessing scaler (optional)  
- `farmer.png` – Display image for UI and README  
- `README.md` – Project documentation (this file)

---

## ▶️ How to Run the App

```bash
# Step 1: Navigate to project folder
cd path/to/project-folder

# Step 2: Run the Streamlit app
streamlit run app.py
