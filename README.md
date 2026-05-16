# Farmer's Credit Worthiness
 
A machine learning project that predicts the creditworthiness of farmers using an XGBoost classifier, complete with a fully automated CI/CD pipeline that retrains, evaluates, and deploys the model on every code push.
 
🌐 **Live App:** [Farmers' Credit Compass on Hugging Face](https://huggingface.co/spaces/Uthman89/farmers_credit)
 
---
 
## Overview
 
Access to credit is a major barrier for smallholder farmers. This project builds a credit scoring model that helps financial institutions assess whether a farmer is likely to repay a loan, based on personal, farm, and financial data. The model is served through an interactive Gradio web app deployed on Hugging Face Spaces.
 
---
 
## Features
 
- XGBoost classifier trained on farmer financial and demographic data
- Automated retraining pipeline triggered on every push to `main`
- Model evaluation report posted under each GitHub commit via CML
- Gradio web app for real-time credit scoring
- Full CI/CD pipeline using GitHub Actions and Hugging Face CLI
---
 
## Project Structure
 
```
farmer-s-credit-worthiness/
├── App/
│   ├── apps.py                  # Gradio web application
│   ├── farmer.png               # App banner image
│   └── requirements.txt         # App dependencies
├── Model/
│   └── model.pkl                # Trained XGBoost model
├── Results/
│   ├── metrics.txt              # Accuracy and F1 score
│   └── model_results.png        # Confusion matrix
├── .github/
│   └── workflows/
│       ├── ci.yml               # Continuous Integration workflow
│       └── cd.yml               # Continuous Deployment workflow
├── train.py                     # Model training and evaluation script
├── Makefile                     # Pipeline command shortcuts
├── requirements.txt             # Project dependencies
└── cleaned_farmers_fin_data.csv # Training dataset
```
 
---
 
## CI/CD Pipeline
 
This project uses a fully automated CI/CD pipeline so that every push to `main` triggers the following sequence without any manual steps:
 
```
Push to main
    └── CI Workflow
        ├── Install dependencies
        ├── Train XGBoost model
        ├── Evaluate and generate metrics + confusion matrix
        ├── Post report under the commit (via CML)
        └── Save model and results to update branch
            └── CD Workflow
                ├── Pull latest model from update branch
                ├── Authenticate with Hugging Face
                └── Deploy app and model to Hugging Face Space
```
 
### Tools Used
 
| Tool | Role |
|---|---|
| GitHub Actions | Workflow automation |
| CML (Iterative) | Posts model metrics under commits |
| Makefile | Command shortcuts for pipeline steps |
| Hugging Face Hub | Model and app hosting |
| XGBoost | Classification model |
| Gradio | Interactive web app framework |
 
---
 
## Model
 
**Algorithm:** XGBoost Classifier
 
**Input Features:**
- Age, education level, marital status, number of dependants
- Farm type, farm size, years of experience
- Access to irrigation and extension services
- Mobile phone access, bank account status
- Previous loan history, loan amount, repayment status
- Total annual income, cooperative membership
**Engineered Features:**
- Farm size × irrigation access
- Income × years of experience
- Log income, square root of loan amount
- Age squared, experience cubed
**Output:** Creditworthy / Not Creditworthy with confidence score
 
---
 
## Getting Started
 
### 1. Clone the repository
 
```bash
git clone https://github.com/Ultra123-hub/farmer-s-credit-worthiness.git
cd farmer-s-credit-worthiness
```
 
### 2. Install dependencies
 
```bash
pip install -r requirements.txt
```
 
### 3. Train the model
 
```bash
python train.py
```
 
### 4. Run the app locally
 
```bash
cd App
python apps.py
```
 
---
 
## GitHub Secrets Required
 
To run the CI/CD pipeline, add these secrets to your GitHub repository under **Settings → Secrets and variables → Actions:**
 
| Secret | Description |
|---|---|
| `HF` | Hugging Face API token (write access) |
| `USER_NAME` | Your GitHub username |
| `USER_EMAIL` | Your GitHub email address |
 
---
 
## Live Demo
 
Try the app here: [https://huggingface.co/spaces/Uthman89/farmers_credit](https://huggingface.co/spaces/Uthman89/farmers_credit)
 
Fill in a farmer's details and click **Assess Creditworthiness** to get an instant prediction with a confidence score.
 
---
 
## License
 
This project is licensed under the MIT License.
