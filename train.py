# importing the necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
import os
import matplotlib.pyplot as plt
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

1. # loading the dataset
df=pd.read_csv('Data/cleaned_farmers_fin_data (1).csv')
# checking the first 5 rows of the dataset
print(df.head())


# ── 2. Encode Target (Yes/No → 1/0) ──────────────────────────
le_target = LabelEncoder()
df['creditworthy'] = le_target.fit_transform(df['creditworthy'])   # No→0, Yes→1

# ── 3. Feature Engineering ────────────────────────────────────
df['farm_size_irrigation'] = df['farm_size_acres'] * df['access_to_irrigation'].astype(int)
df['income_experience']    = df['total_annual_income'] * df['years_of_experience']
df['log_income']           = np.log1p(df['total_annual_income'])
df['sqrt_loan_amount']     = np.sqrt(df['previous_loan_amount'])
df['age_squared']          = df['age'] ** 2
df['experience_cubed']     = df['years_of_experience'] ** 3

# ── 4. Encode Categoricals ─────────────────────────────────────
categorical_cols = [
    'education_level', 'marital_status', 'farm_type',
    'access_to_extension_services', 'previous_loan_history', 'group_membership',
    # Additional string columns present in this dataset
    'gender', 'location_state', 'other_income_sources'
]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
df.drop(columns=['farmer_id'], errors='ignore', inplace=True)

# ── 5. Split ───────────────────────────────────────────────────
target = 'creditworthy'
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save feature names for the app
joblib.dump(X_train.columns.tolist(), "feature_names.pkl")

# ── 6. Train ───────────────────────────────────────────────────
model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
model.fit(X_train, y_train)
os.makedirs("Model", exist_ok=True)
joblib.dump(model, "Model/model.pkl")

# ── 7. Evaluate ────────────────────────────────────────────────
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, average='macro')

os.makedirs("Results", exist_ok=True)
with open("Results/metrics.txt", "w") as f:
    f.write(f"Accuracy = {round(accuracy, 4)}\n")
    f.write(f"F1 Score = {round(f1, 4)}\n")
    f.write(f"\n{classification_report(y_test, preds)}")

# ── 8. Confusion Matrix ────────────────────────────────────────
cm = confusion_matrix(y_test, preds)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("Results/model_results.png", dpi=120)

print(f"✅ Training complete | Accuracy: {round(accuracy*100, 2)}% | F1: {round(f1, 4)}")
