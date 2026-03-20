# 🫀 Heart Disease Prediction — End to End ML Product

## ❗ Problem Statement

Heart disease is the **#1 cause of death globally** — killing 17.9 million people every year. The core problem isn't just medical. It's economic and systemic:

- Late-stage heart disease treatment costs **$1M+ per patient**
- Early detection reduces that to **~$10,000**
- That's a **100x cost difference**
- Clinics in developing countries lack fast, affordable screening tools
- Specialists are expensive, waiting lists are long, patients arrive too late

This product addresses that gap — allowing anyone to input basic clinical vitals and instantly receive a risk prediction with full explainability, making early screening faster, accessible, and data-driven.

---

## 🌍 Live Demo

👉 **[https://heart-disease-prediction-e2e.onrender.com](https://heart-disease-prediction-e2e.onrender.com)**

---

## 💡 Use Case & ROI

| Scenario | Without Tool | With Tool |
|---|---|---|
| Screening time | Hours (specialist needed) | < 2 seconds |
| Cost per screening | $200–$500 | Near zero |
| Accessibility | Hospital only | Any device, anywhere |
| Explainability | Doctor's intuition | SHAP-driven feature impact |
| Detection threshold | Standard 0.5 | Tuned to 0.3 (recall-optimized) |

**Target users:**
- Rural clinics with no specialist access
- General practitioners doing quick triage
- Patients who want to understand their own risk factors
- Researchers studying cardiovascular risk patterns

---

## 📁 Project Structure

```
Heart Disease Prediction E2E/
├── data/
│   ├── heart_cleveland_upload.csv   ← Cleveland dataset
│   ├── shap_background.csv          ← Background data for SHAP
│   └── predictions_log.csv          ← Auto-generated prediction logs
├── model/
│   └── heart_disease_model.pkl      ← Trained model
├── notebook/
│   └── notebook.ipynb               ← EDA, training, evaluation
├── src/
│   ├── backend/
│   │   └── main.py                  ← FastAPI app
│   └── frontend/
│       └── app.py                   ← Streamlit dashboard
├── .github/
│   └── workflows/
│       └── deploy.yml               ← CI/CD pipeline
├── Dockerfile
├── start.sh
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

> **Why Recall over Accuracy?** In medicine, a false negative is dangerous — missing a sick patient is worse than flagging a healthy one. Tuning from 0.5 → 0.3 catches more at-risk patients even at the cost of more false alarms.

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
- **Prediction Logging** — Every prediction saved to CSV automatically
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
| Containerization | Docker |
| Deployment | Render |
| CI/CD | GitHub Actions |

---

## 💪 Strengths

- **Recall-optimized** — catches more sick patients, medically justified decision
- **Explainable AI** — SHAP shows exactly why each prediction was made
- **Full MLOps pipeline** — Docker + CI/CD + logging + live deployment
- **Human-friendly** — plain English summaries, actionable next steps
- **Responsible AI** — medical disclaimer, confidence levels, not just binary output

---

## ⚠️ Limitations & Honest Notes

- **Small dataset** — 303 samples is not production scale. Model performance may not generalize to all populations
- **CSV logging** — used for simplicity as this is a first MLOps project. A production system would use a proper database (PostgreSQL, SQLite) with timestamps, user tracking, and drift monitoring
- **No authentication** — anyone can hit the API. Production would require API keys or OAuth
- **No model monitoring** — no automated alerts if model performance degrades over time
- **Not a medical device** — this is a screening tool and prototype, not a certified clinical diagnostic tool. Always consult a qualified cardiologist

---

## 🚀 How to Run

### Option A — Live Demo
👉 **[https://heart-disease-prediction-e2e.onrender.com](https://heart-disease-prediction-e2e.onrender.com)**

### Option B — Docker
```bash
docker build -t heart-disease .
docker run -p 8000:8000 -p 8001:8001 heart-disease
```
Open `http://localhost:8000`

### Option C — Local
```bash
# Install dependencies
pip install -r requirements.txt

# Terminal 1 — FastAPI
cd src/backend
uvicorn main:app --host 0.0.0.0 --port 8001 --reload

# Terminal 2 — Streamlit
cd src/frontend
streamlit run app.py
```

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

## 👤 Author

**Mohammad Naif** — Cool Data Science Undergrad Student  
Building towards a career in Agentic AI Engineering 🚀
