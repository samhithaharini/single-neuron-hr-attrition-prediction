import streamlit as st
import numpy as np
import pandas as pd
import pickle

model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.title("HR Employee Attrition Prediction")

satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
last_evaluation = st.slider("Last Evaluation", 0.0, 1.0, 0.5)
number_project = st.number_input("Number of Projects", 1, 10, 3)
average_montly_hours = st.number_input("Monthly Hours", 100, 350, 200)
time_spend_company = st.number_input("Years in Company", 1, 10, 3)
Work_accident = st.selectbox("Work Accident", [0,1])
promotion_last_5years = st.selectbox("Promotion Last 5 Years", [0,1])

Department = st.selectbox("Department", 
    ["sales","technical","support","IT","hr","product_mng","marketing","accounting","management"])

salary = st.selectbox("Salary", ["low","medium","high"])

input_dict = {
    'satisfaction_level': satisfaction_level,
    'last_evaluation': last_evaluation,
    'number_project': number_project,
    'average_montly_hours': average_montly_hours,
    'time_spend_company': time_spend_company,
    'Work_accident': Work_accident,
    'promotion_last_5years': promotion_last_5years,
    'Department_' + Department: 1,
    'salary_' + salary: 1
}

input_df = pd.DataFrame([input_dict])

input_df = input_df.reindex(columns=columns, fill_value=0)

if st.button("Predict"):

    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"Employee will LEAVE ❌ (Prob: {prob:.2f})")
    else:
        st.success(f"Employee will STAY ✅ (Prob: {1-prob:.2f})")