# 🫀 Heart Disease Prediction — End to End ML Product

## ❗ Problem Statement

Heart disease is one of the leading causes of death globally. Early detection is critical but often requires expensive clinical tests and specialist interpretation. This product allows anyone to input basic patient vitals and instantly receive a risk prediction — making early screening faster, accessible, and data-driven.

---

## 📁 Project Structure

```
Heart Disease Prediction E2E/
├── data/
│   ├── heart_cleveland_upload.csv   ← Cleveland dataset
│   └── shap_background.csv          ← Background data for SHAP
├── model/
│   └── heart_disease_model.pkl      ← Trained model
├── notebook/
│   └── notebook.ipynb               ← EDA, training, evaluation
├── src/
│   ├── backend/
│   │   └── main.py                  ← FastAPI app
│   └── frontend/
│       └── app.py                   ← Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## 🧠 ML Pipeline

- **Dataset:** Cleveland Heart Disease Dataset (303 samples, 13 features)
- **Models Compared:** Logistic Regression, Random Forest, XGBoost
- **Best Model:** Logistic Regression — selected based on Recall to minimize missed diagnoses
- **Threshold Tuning:** Custom threshold of **0.3** (vs default 0.5) — optimized for Recall over Precision
- **Explainability:** SHAP KernelExplainer — feature impact per prediction
- **Serialization:** joblib

**Input Features:**

| Feature | Description |
|---|---|
| age | Age of patient |
| sex | 0 = Female, 1 = Male |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol |
| fbs | Fasting blood sugar |
| restecg | Resting ECG results |
| thalach | Max heart rate achieved |
| exang | Exercise induced angina |
| oldpeak | ST depression |
| slope | Slope of peak exercise ST |
| ca | Major vessels colored (0–3) |
| thal | Thalassemia (0–3) |

---

## 🎨 Dashboard Features

- **Confidence Levels** — Borderline / Moderate / High Risk banners
- **Risk Gauge Meter** — Visual probability indicator (0–100%)
- **SHAP Explainability** — Bar chart showing why the model made its prediction
- **Plain English Summary** — Top risk factors explained in simple language
- **Recommended Next Steps** — Actionable advice based on risk level
- **Medical Disclaimer** — Responsible AI disclosure
- **About Section** — Tool description and usage guide

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML | scikit-learn, Logistic Regression |
| Explainability | SHAP |
| API | FastAPI, Pydantic, Uvicorn |
| Dashboard | Streamlit, Plotly |
| Data | Pandas, NumPy |
| Serialization | Joblib |

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start FastAPI backend (Terminal 1)
```bash
cd src/backend
uvicorn main:app --reload
```

### 3. Start Streamlit frontend (Terminal 2)
```bash
cd src/frontend
streamlit run app.py
```

### 4. Open in browser
- **Streamlit Dashboard** → `http://localhost:8501`
- **FastAPI Swagger Docs** → `http://localhost:8000/docs`

> ⚠️ Both terminals must be running at the same time.

---

## 📊 Sample Prediction

**Input:**
```json
{
  "age": 63, "sex": 1, "cp": 3,
  "trestbps": 145, "chol": 233,
  "fbs": 1, "restecg": 2,
  "thalach": 150, "exang": 0,
  "oldpeak": 2.3, "slope": 0,
  "ca": 0, "thal": 1
}
```

**Output:**
```json
{
  "heart_disease_prediction": 1,
  "disease_probability": 0.5998,
  "result": "Heart Disease Detected",
  "shap_values": {
    "cp": 0.147,
    "oldpeak": 0.136,
    "restecg": 0.043,
    "ca": -0.093
  }
}
```

---

## ✅ Completed Features
- [x] SHAP explainability — feature impact per prediction
- [x] Risk meter gauge chart
- [x] Confidence levels — Borderline / Moderate / High Risk
- [x] Plain English summary
- [x] Recommended next steps per risk level
- [x] Medical disclaimer
- [x] Custom threshold (0.3) for recall optimization

## 🔮 Planned Features — MLOps Roadmap
- [ ] Prediction logging — CSV log of every prediction made
- [ ] Docker containerization
- [ ] Cloud deployment (Render)
- [ ] CI/CD pipeline — GitHub Actions auto deploy
- [ ] MLflow model registry — experiment tracking
- [ ] Normal range warnings per input feature
- [ ] Model performance monitoring

---

## 👤 Author

**Mohammad Naif** — Cool Data Science Undergrad Student  
Building towards a career in Agentic AI Engineering 🚀
