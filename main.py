import pandas as pd
from src.data_loader import load_data
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# -----------------------------
# 1. Load Data
# -----------------------------
data = load_data()

# Select required columns
data = data[[
    'Age',
    'JobRole',
    'MonthlyIncome',
    'YearsAtCompany',
    'TrainingTimesLastYear'
]]

# Rename columns
data.columns = [
    'Age',
    'Department',
    'Salary',
    'Experience',
    'TrainingHours'
]

# -----------------------------
# 2. 🔥 CREATE NEW PERFORMANCE TARGET (FINAL FIX)
# -----------------------------
score = (
    data['Experience'] * 0.4 +
    data['TrainingHours'] * 0.3 +
    (data['Salary'] / 10000) * 0.3
)

# Convert score into categories
data['Performance'] = pd.cut(
    score,
    bins=3,
    labels=['Low', 'Medium', 'High']
)

print("\n✅ New Performance Distribution:")
print(data['Performance'].value_counts())

# -----------------------------
# 3. Encoding
# -----------------------------
le_dept = LabelEncoder()
le_perf = LabelEncoder()

data['Department'] = le_dept.fit_transform(data['Department'])
data['Performance'] = le_perf.fit_transform(data['Performance'])

# Features & Target
X = data.drop('Performance', axis=1)
y = data['Performance']

# -----------------------------
# 4. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 5. Model Training (Improved)
# -----------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# 6. Evaluation
# -----------------------------
y_pred = model.predict(X_test)

print("\n✅ Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# -----------------------------
# 7. Save Model
# -----------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/model.pkl")
joblib.dump(le_dept, "models/le_dept.pkl")
joblib.dump(le_perf, "models/le_perf.pkl")

print("\n✅ Model saved successfully!")