import streamlit as st
import requests 
import plotly.graph_objects as go


# Feature name mapping
FEATURE_NAMES = {
    "age":      "Age",
    "sex":      "Gender",
    "cp":       "Chest Pain Type",
    "trestbps": "Resting Blood Pressure",
    "chol":     "Serum Cholesterol",
    "fbs":      "Fasting Blood Sugar",
    "restecg":  "Resting ECG Results",
    "thalach":  "Maximum Heart Rate",
    "exang":    "Exercise Induced Angina",
    "oldpeak":  "ST Depression",
    "slope":    "ST Slope",
    "ca":       "Blocked Vessels",
    "thal":     "Thalassemia"
}

st.title("🫀 Heart Disease Predictor")
st.markdown("Fill in the patient details below and click **Predict** to get the result.")

with st.expander("ℹ️  About this tool"):
    st.markdown("""
    ### What is this?
    An AI-powered screening tool that predicts heart disease risk 
    based on clinical parameters from the Cleveland Heart Disease dataset.
    
    ### Who is it for?
    - Healthcare professionals for quick preliminary screening
    - Patients who want to understand their risk factors
    - Researchers studying cardiovascular risk
    
    ### How does it work?
    1. Fill in the patient's clinical details
    2. Click **Predict**
    3. Get instant risk score + explanation of key risk factors
    
    ### Important
    - Model trained on Cleveland Heart Disease Dataset (303 patients)
    - Optimized for **Recall** — minimizes missed diagnoses
    - Detection threshold set at **0.3** (sensitive screening)
    """)


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


        st.metric(label="Disease Probability", value=f"{result['disease_probability'] * 100:.2f}%")
        
        shap_data = result["shap_values"]
        
        # Finding top 2 risk factors (highest positive SHAP values
        risk_factors = {k: v for k, v in shap_data.items() if v > 0}
        top_2 = sorted(risk_factors, key=risk_factors.get, reverse=True)[:2]
        top_2_plain = [FEATURE_NAMES.get(f, f) for f in top_2]
        
        prob = result["disease_probability"] * 100

        if result["heart_disease_prediction"] == 1:
            if len(top_2_plain) >= 2:
                summary = f"⚠️ **{top_2_plain[0]}** and **{top_2_plain[1]}** are the biggest contributors to this patient's risk score."
            elif len(top_2_plain) == 1:
                summary = f"⚠️ **{top_2_plain[0]}** is the biggest contributor to this patient's risk score."
            else:
                summary = "⚠️ Multiple factors are contributing to this patient's risk score."
        else:
            summary = "✅ No major risk factors detected. Patient shows healthy clinical indicators."

        st.info(summary)

        
        prob = result["disease_probability"] * 100
        if prob < 30:
            risk_level = "Low Risk"
            color = "green"
        elif prob < 60:
            risk_level = "Medium Risk"
            color = "orange"
        else:
            risk_level = "High Risk"
            color = "red"
            
            
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            number={"suffix": "%"},
            title={"text": f"Risk Level: {risk_level}"},
            gauge={
                "axis" : {"range" : [0,100]},
                "bar" : {"color": color},
                "steps" : [
                    {"range": [0, 30],   "color": "#d4edda"},
                    {"range": [30, 60],  "color": "#fff3cd"},
                    {"range": [60, 100], "color": "#f8d7da"},
                ],
            }
        ))
        
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        
        st.subheader("🔍 Why this prediction?")
        
        shap_data = result["shap_values"]
        
        # Sorting by absolute impact 
        sorted_shap = sorted(shap_data.items(), key=lambda x: abs(x[1]))
        features = [FEATURE_NAMES.get(str(k), str(k)) for k, v in sorted_shap]
        values = [float(v) for k, v in sorted_shap]
        colors = ["red" if v > 0 else "green" for v in values]
        
        fig_shap = go.Figure(go.Bar(
            x=values,
            y=features,
            orientation="h",
            marker_color=colors
        ))
        
        fig_shap.update_layout(
            xaxis_title="Impact on Prediction",
            yaxis_title = "Feature",
            height = 400
        )
        
        st.plotly_chart(fig_shap, use_container_width=True)
        st.caption("🔴 Red = increases heart disease risk  |  🟢 Green = decreases risk")
        
        
        st.warning("""
        **⚠️ Medical Disclaimer**  
        This tool is for **screening purposes only** and does not constitute a medical diagnosis.  
        Please consult a qualified cardiologist for proper clinical evaluation and treatment.
        """)
        
        if result["heart_disease_prediction"] == 1:
            st.subheader("📋 Recommended Next Steps")
            prob = result["disease_probability"] * 100
            if prob < 45:
                st.markdown("""
                - 🩺 Schedule a routine checkup with your GP
                - 📊 Request a lipid panel and ECG test
                - 🏃 Review lifestyle factors (diet, exercise, smoking)
                """)
            elif prob < 70:
                st.markdown("""
                - 🏥 Visit a cardiologist within the next 2-4 weeks
                - 🩻 Request a stress test and echocardiogram
                - 💊 Discuss medication options with your doctor
                - 🚭 Immediately address lifestyle risk factors
                """)
            else:
                st.markdown("""
                - 🚨 Seek immediate medical attention
                - 📞 Call your cardiologist today
                - 🏥 Consider emergency cardiac evaluation
                - ❌ Avoid strenuous physical activity until evaluated
                """)

    except Exception as e:
        st.error(f"API Error: {str(e)}")