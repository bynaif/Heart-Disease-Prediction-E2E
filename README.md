# 🫀 Heart Disease Prediction — End to End ML Product

## ❗ Problem Statement

Heart disease is one of the leading causes of death globally. Early detection is critical but often requires expensive clinical tests and specialist interpretation. This product allows anyone to input basic patient vitals and instantly receive a risk prediction — making early screening faster, accessible, and data-driven.

---

## 📁 Project Structure

```
Heart Disease Prediction E2E/
├── data/
│   └── heart_cleveland_upload.csv   ← Cleveland dataset
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
- **Best Model:** XGBoost — selected based on Recall to minimize missed diagnoses
- **Threshold Tuning:** Optimized for recall over precision
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

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML | scikit-learn, XGBoost |
| API | FastAPI, Pydantic, Uvicorn |
| Dashboard | Streamlit |
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
  "result": "Heart Disease Detected"
}
```

---

## 🔮 Planned Features

- [ ] SHAP explainability — feature impact per prediction
- [ ] Risk meter gauge chart
- [ ] Prediction history logging
- [ ] Docker containerization
- [ ] Cloud deployment (Render / Railway)

---

## 👤 Author

**Mohammad Naif** — Cool Data Science Undergrad Student  
