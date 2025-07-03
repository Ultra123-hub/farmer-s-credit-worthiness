import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import streamlit as st
from PIL import Image

# Title
st.title("Farmers' Credit Compass")

image = Image.open("farmer.png")  # Replace with your image name
st.image(image, caption="An old farmer's picture with cash at hand (generated with AI)", use_container_width=True)


st.markdown(
    """
    <div style="text-align: center;">
        <img src='farmer.png' width='300'>
    </div>
    """,
    unsafe_allow_html=True
)


# Load model
model = xgb.XGBClassifier()
model.load_model("farmer_optimized_xgb.json")  # Ensure correct path

# Load scaler
scaler = joblib.load("model.pkl")  # Ensure correct path


st.write("Fill in the farmer's details to assess their creditworthiness.")



# Input form
age = st.number_input("Age", min_value=18, max_value=100, value=30)

education_level = st.selectbox("Education Level", ["None", "Primary", "Secondary", "Tertiary"])

marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])

dependants = st.number_input("Number of Dependants", min_value=0, value=2)

farm_type = st.selectbox("Farm Type", ["Crop", "Livestock", "Mixed"])

farm_size_acres = st.number_input("Farm Size (in acres)", min_value=0.0, value=2.5)

years_of_experience = st.number_input("Years of Farming Experience", min_value=0, value=5)

access_to_extension_services = st.checkbox("Access to Extension Services", value=True)
access_to_irrigation = st.checkbox("Access to Irrigation", value=False)
mobile_phone_access = st.checkbox("Has Mobile Phone Access", value=True)
has_bank_account = st.checkbox("Has Bank Account", value=True)
previous_loan_history = st.checkbox("Has Taken Previous Loan", value=False)
previous_loan_amount = st.number_input("Previous Loan Amount (₦)", min_value=0.0, value=0.0)
previous_loan_repaid = st.checkbox("Previous Loan Fully Repaid", value=False)
total_annual_income = st.number_input("Total Annual Income (₦)", min_value=0, value=500000)
group_membership = st.checkbox("Member of Farmer Group/Cooperative", value=True)

# Encode categorical variables manually
education_dict = {"None": 0, "Primary": 1, "Secondary": 2, "Tertiary": 3}
marital_dict = {"Single": 0, "Married": 1, "Divorced": 2, "Widowed": 3}
farm_type_dict = {"Crop": 0, "Livestock": 1, "Mixed": 2}

# Create input DataFrame
input_data = pd.DataFrame([{
    'age': age,
    'education_level': education_dict[education_level],
    'marital_status': marital_dict[marital_status],
    'dependants': dependants,
    'farm_type': farm_type_dict[farm_type],
    'farm_size_acres': farm_size_acres,
    'years_of_experience': years_of_experience,
    'access_to_extension_services': int(access_to_extension_services),
    'access_to_irrigation': int(access_to_irrigation),
    'mobile_phone_access': int(mobile_phone_access),
    'has_bank_account': int(has_bank_account),
    'previous_loan_history': int(previous_loan_history),
    'previous_loan_amount': previous_loan_amount,
    'previous_loan_repaid': int(previous_loan_repaid),
    'total_annual_income': total_annual_income,
    'group_membership': int(group_membership)
}])



# Add engineered features first
input_data['farm_size_irrigation'] = input_data['farm_size_acres'] * input_data['access_to_irrigation']
input_data['income_experience'] = input_data['total_annual_income'] * input_data['years_of_experience']
input_data['log_income'] = np.log1p(input_data['total_annual_income'])
input_data['sqrt_loan_amount'] = np.sqrt(input_data['previous_loan_amount'])
input_data['age_squared'] = input_data['age']**2
input_data['experience_cubed'] = input_data['years_of_experience']**3


model_feature_names = joblib.load("feature_names.pkl")
# Create one-hot encoded columns consistent with training
categorical_cols = ['education_level', 'marital_status', 'farm_type', 
                       'access_to_extension_services', 'previous_loan_history', 
                       'group_membership']
# First, get all non-encoded columns
non_encoded_cols = [col for col in input_data.columns if col not in categorical_cols]
encoded_part = pd.get_dummies(input_data[categorical_cols], drop_first=True)
    
# Combine non-encoded with encoded
temp_input = pd.concat([input_data[non_encoded_cols], encoded_part], axis=1)
    
# Create final DataFrame with exact columns the model expects
encoded_input = pd.DataFrame(index=input_data.index)

# Copy over features that exist in our current data
for col in model_feature_names:
    if col in temp_input.columns:
        encoded_input[col] = temp_input[col]
    else:
# For missing columns (potential categories not in current data), add zeros
        encoded_input[col] = 0

# Drop identifier columns before prediction
encoded_input = encoded_input.drop(columns=['farmer_id'], errors='ignore')

# Predict
prediction = model.predict(encoded_input)[0]


# Display result
if prediction == 1:
    st.success("✅ The farmer is predicted to be **creditworthy**.")
else:
    st.error("❌ The farmer is predicted not to be **creditworthy**.")




