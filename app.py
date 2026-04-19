import streamlit as st
import pandas as pd
import joblib
from src.data_loader import load_data

# Load model
model = joblib.load("models/model.pkl")
le_dept = joblib.load("models/le_dept.pkl")
le_perf = joblib.load("models/le_perf.pkl")

st.set_page_config(page_title="Employee Performance Predictor", layout="wide")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "Dashboard"])

# -----------------------------
# Prediction Page
# -----------------------------
if page == "Prediction":

    st.title("👨‍💼 Employee Performance Predictor")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 18, 60, 30)
        experience = st.slider("Experience", 0, 40, 5)

    with col2:
        department = st.selectbox("Department", list(le_dept.classes_))
        training = st.slider("Training Hours", 0, 50, 10)

    with col3:
        salary = st.number_input("Salary", 1000, 200000, 50000)

    if st.button("Predict"):

        dept_encoded = le_dept.transform([department])[0]

        # 🔥 FIXED INPUT FORMAT
        input_df = pd.DataFrame({
            'Age': [age],
            'Department': [dept_encoded],
            'Salary': [salary],
            'Experience': [experience],
            'TrainingHours': [training]
        })

        prediction = model.predict(input_df)
        result = le_perf.inverse_transform(prediction)

        # Debug (remove later)
        st.write("Input Data:", input_df)
        st.write("Raw Prediction:", prediction)

        st.success(f"Predicted Performance: {result[0]}")

        if result[0] == "High":
            st.balloons()
        elif result[0] == "Low":
            st.warning("⚠ Needs Improvement")

# -----------------------------
# Dashboard Page
# -----------------------------
elif page == "Dashboard":

    st.title("📊 HR Dashboard")

    data = load_data()

    st.subheader("Performance Distribution (Raw Data)")
    st.bar_chart(data['PerformanceRating'].value_counts())