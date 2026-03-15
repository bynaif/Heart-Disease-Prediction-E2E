import streamlit as st
import requests 

st.title("🫀 Heart Disease Predictor")
st.markdown("Fill in the patient details below and click **Predict** to get the result.")

st.subheader("Patient Details")

col1, col2 = st.columns(2)

with col1:
    age      = st.number_input("Age", min_value=29, max_value=77)
    sex      = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    cp       = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], format_func=lambda x: {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal Pain", 3: "Asymptomatic"}[x])
    trestbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=94, max_value=200)
    chol     = st.number_input("Serum Cholesterol (mg/dl)", min_value=126, max_value=564)
    fbs      = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "True" if x == 1 else "False")
    restecg  = st.selectbox("Resting ECG Results", options=[0, 1, 2], format_func=lambda x: {0: "Normal", 1: "ST-T Abnormality", 2: "Left Ventricular Hypertrophy"}[x])

with col2:
    thalach  = st.number_input("Max Heart Rate Achieved", min_value=71, max_value=202)
    exang    = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    oldpeak  = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=6.2, step=0.1, format="%.1f")
    slope    = st.selectbox("Slope of Peak Exercise ST", options=[0, 1, 2], format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x])
    ca       = st.selectbox("Major Vessels Colored (0–3)", options=[0, 1, 2, 3])
    thal     = st.selectbox("Thalassemia", options=[0, 1, 2, 3], format_func=lambda x: {0: "Normal", 1: "Fixed Defect", 2: "Reversible Defect", 3: "Unknown"}[x])


if st.button("🔍 Predict"):
    payload = {
        "age": int(age), "sex": int(sex), "cp": int(cp),
        "trestbps": int(trestbps), "chol": int(chol), "fbs": int(fbs),
        "restecg": int(restecg), "thalach": int(thalach), "exang": int(exang),
        "oldpeak": float(oldpeak), "slope": int(slope), "ca": int(ca), "thal": int(thal)
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        result   = response.json()

        st.divider()
        st.subheader("Prediction Result")

        if result["heart_disease_prediction"] == 1:
            st.error(f"❤️‍🩹 {result['result']}")
        else:
            st.success(f"💚 {result['result']}")

        st.metric(label="Disease Probability", value=f"{result['disease_probability'] * 100:.2f}%")

    except Exception as e:
        st.error(f"API Error: {str(e)}")