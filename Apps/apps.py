import gradio as gr
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

# Load model and feature names
model = xgb.XGBClassifier()
model.load_model("farmer_optimized_xgb.json")
model_feature_names = joblib.load("feature_names.pkl")

def predict_creditworthiness(
    age, education_level, marital_status, dependants, farm_type,
    farm_size_acres, years_of_experience, access_to_extension_services,
    access_to_irrigation, mobile_phone_access, has_bank_account,
    previous_loan_history, previous_loan_amount, previous_loan_repaid,
    total_annual_income, group_membership
):
    education_dict = {"None": 0, "Primary": 1, "Secondary": 2, "Tertiary": 3}
    marital_dict   = {"Single": 0, "Married": 1, "Divorced": 2, "Widowed": 3}
    farm_type_dict = {"Crop": 0, "Livestock": 1, "Mixed": 2}

    input_data = pd.DataFrame([{
        'age':                          age,
        'education_level':              education_dict[education_level],
        'marital_status':               marital_dict[marital_status],
        'dependants':                   dependants,
        'farm_type':                    farm_type_dict[farm_type],
        'farm_size_acres':              farm_size_acres,
        'years_of_experience':          years_of_experience,
        'access_to_extension_services': int(access_to_extension_services),
        'access_to_irrigation':         int(access_to_irrigation),
        'mobile_phone_access':          int(mobile_phone_access),
        'has_bank_account':             int(has_bank_account),
        'previous_loan_history':        int(previous_loan_history),
        'previous_loan_amount':         previous_loan_amount,
        'previous_loan_repaid':         int(previous_loan_repaid),
        'total_annual_income':          total_annual_income,
        'group_membership':             int(group_membership)
    }])

    # Feature engineering
    input_data['farm_size_irrigation'] = input_data['farm_size_acres'] * input_data['access_to_irrigation']
    input_data['income_experience']    = input_data['total_annual_income'] * input_data['years_of_experience']
    input_data['log_income']           = np.log1p(input_data['total_annual_income'])
    input_data['sqrt_loan_amount']     = np.sqrt(input_data['previous_loan_amount'].clip(lower=0))
    input_data['age_squared']          = input_data['age'] ** 2
    input_data['experience_cubed']     = input_data['years_of_experience'] ** 3

    # Align columns with training
    categorical_cols  = ['education_level', 'marital_status', 'farm_type',
                         'access_to_extension_services', 'previous_loan_history', 'group_membership']
    non_encoded_cols  = [col for col in input_data.columns if col not in categorical_cols]
    encoded_part      = pd.get_dummies(input_data[categorical_cols], drop_first=True)
    temp_input        = pd.concat([input_data[non_encoded_cols], encoded_part], axis=1)

    encoded_input = pd.DataFrame(index=input_data.index)
    for col in model_feature_names:
        encoded_input[col] = temp_input[col] if col in temp_input.columns else 0

    encoded_input.drop(columns=['farmer_id'], errors='ignore', inplace=True)

    prediction    = model.predict(encoded_input)[0]
    probability   = model.predict_proba(encoded_input)[0]
    confidence    = round(float(max(probability)) * 100, 2)

    if prediction == 1:
        return (
            "CREDITWORTHY",
            f"The farmer is predicted to be creditworthy with {confidence}% confidence.",
            confidence / 100
        )
    else:
        return (
            "NOT CREDITWORTHY",
            f"The farmer is predicted not to be creditworthy with {confidence}% confidence.",
            confidence / 100
        )


# ── UI Layout ─────────────────────────────────────────────────────────
with gr.Blocks(title="Farmers' Credit Compass", theme=gr.themes.Soft()) as app:

    gr.Markdown("# Farmers' Credit Compass")
    gr.Markdown("Fill in the farmer's details below to assess their creditworthiness.")

    with gr.Row():
        # Left column — personal details
        with gr.Column():
            gr.Markdown("### Personal Details")
            age               = gr.Number(label="Age", value=30, minimum=18, maximum=100)
            education_level   = gr.Dropdown(label="Education Level",
                                            choices=["None", "Primary", "Secondary", "Tertiary"],
                                            value="Primary")
            marital_status    = gr.Dropdown(label="Marital Status",
                                            choices=["Single", "Married", "Divorced", "Widowed"],
                                            value="Married")
            dependants        = gr.Number(label="Number of Dependants", value=2, minimum=0)
            mobile_phone_access = gr.Checkbox(label="Has Mobile Phone Access", value=True)
            has_bank_account    = gr.Checkbox(label="Has Bank Account",         value=True)
            group_membership    = gr.Checkbox(label="Member of Farmer Group / Cooperative", value=True)

        # Middle column — farm details
        with gr.Column():
            gr.Markdown("### Farm Details")
            farm_type              = gr.Dropdown(label="Farm Type",
                                                 choices=["Crop", "Livestock", "Mixed"],
                                                 value="Crop")
            farm_size_acres        = gr.Number(label="Farm Size (acres)",            value=2.5, minimum=0)
            years_of_experience    = gr.Number(label="Years of Farming Experience",  value=5,   minimum=0)
            access_to_extension_services = gr.Checkbox(label="Access to Extension Services", value=True)
            access_to_irrigation         = gr.Checkbox(label="Access to Irrigation",         value=False)
            total_annual_income          = gr.Number(label="Total Annual Income (NGN)",
                                                     value=500000, minimum=0)

        # Right column — loan history
        with gr.Column():
            gr.Markdown("### Loan History")
            previous_loan_history = gr.Checkbox(label="Has Taken a Previous Loan",      value=False)
            previous_loan_amount  = gr.Number(label="Previous Loan Amount (NGN)",
                                              value=0.0, minimum=0)
            previous_loan_repaid  = gr.Checkbox(label="Previous Loan Fully Repaid",     value=False)

    # Predict button
    predict_btn = gr.Button("Assess Creditworthiness", variant="primary", size="lg")

    gr.Markdown("---")
    gr.Markdown("### Prediction Result")

    with gr.Row():
        verdict     = gr.Textbox(label="Verdict",     interactive=False)
        explanation = gr.Textbox(label="Explanation", interactive=False)

    confidence_bar = gr.Slider(label="Confidence Score", minimum=0, maximum=1,
                               interactive=False, value=0)

    predict_btn.click(
        fn=predict_creditworthiness,
        inputs=[
            age, education_level, marital_status, dependants, farm_type,
            farm_size_acres, years_of_experience, access_to_extension_services,
            access_to_irrigation, mobile_phone_access, has_bank_account,
            previous_loan_history, previous_loan_amount, previous_loan_repaid,
            total_annual_income, group_membership
        ],
        outputs=[verdict, explanation, confidence_bar]
    )

    gr.Markdown("---")
    gr.Markdown("*Farmers' Credit Compass — powered by XGBoost*")

app.launch()
