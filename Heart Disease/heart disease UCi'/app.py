import streamlit as st
import joblib
import numpy as np
import sqlite3
import pandas as pd
from datetime import datetime

# databse nako
def init_db():
    conn = sqlite3.connect('heart_history.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS medical_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            age INTEGER, sex TEXT, cp TEXT, trestbps INTEGER, 
            chol INTEGER, fbs TEXT, restecg TEXT, thalach INTEGER, 
            exang TEXT, oldpeak REAL, slope TEXT, ca INTEGER, 
            thal TEXT, prediction TEXT, probability REAL
        )
    ''')
    conn.commit()
    conn.close()

def add_history(data_tuple):
    conn = sqlite3.connect('heart_history.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO medical_history (
            timestamp, age, sex, cp, trestbps, chol, fbs, restecg, 
            thalach, exang, oldpeak, slope, ca, thal, prediction, probability
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    ''', data_tuple)
    conn.commit()
    conn.close()

# Initialize database
init_db()

# streamlit nako 
st.set_page_config(page_title="Heart Health", page_icon="❤️", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #fff5f5; }
    h1 { color: #c0392b; text-align: center; border-bottom: 2px solid #e74c3c; }
    .stWidgetLabel p { color: #7f8c8d; font-weight: bold; }
    div.stButton > button:first-child {
        background-color: #e74c3c; color: white; font-size: 20px;
        width: 100%; border-radius: 10px; transition: 0.3s;
    }
    div.stButton > button:first-child:hover { background-color: #c0392b; }
    </style>
    """, unsafe_allow_html=True)
# MODEL
@st.cache_resource
def load_model():
    return joblib.load("heart_model.pkl")

try:
    model = load_model()
except:
    st.error("⚠️ Model file not found!")

tab1, tab2 = st.tabs(["🔍 Diagnostic Dashboard", "📜 Patient History"])

with tab1:
    st.markdown("<h1>❤️ Heart Disease Diagnostic Dashboard</h1>", unsafe_allow_html=True)
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("👤 Patient Information")
            age = st.number_input("Patient Age", 1, 120, 45)
            sex_val = st.selectbox("Sex", [(1, "Male"), (0, "Female")], format_func=lambda x: x[1])
            cp_val = st.selectbox("Chest Pain Type", [(0,"Typical Angina"), (1,"Atypical Angina"), (2,"Non-anginal Pain"), (3,"Asymptomatic")])
            trestbps = st.number_input("Resting Blood Pressure", 50, 250, 120)
            chol = st.number_input("Serum Cholesterol", 100, 600, 200)
            fbs_val = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [(1, "True"), (0, "False")], format_func=lambda x: x[1])

        with col2:
            st.subheader("🩺 Clinical Tests")
            restecg_val = st.selectbox("Resting ECG Results", [(0,"Normal"), (1,"ST-T Wave"), (2,"LV Hypertrophy")])
            thalach = st.number_input("Max Heart Rate Achieved", 50, 250, 150)
            exang_val = st.selectbox("Exercise Induced Angina", [(1, "Yes"), (0, "No")], format_func=lambda x: x[1])
            oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 10.0, 1.0)
            slope_val = st.selectbox("Slope of Peak Exercise", [(0,"Upsloping"), (1,"Flat"), (2,"Downsloping")])
            ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
            thal_val = st.selectbox("Thalassemia Status", [(0,"Normal"), (1,"Fixed Defect"), (2,"Reversible Defect")])

    if st.button("🔍 ANALYZE HEART HEALTH"):
       # DATA NAKO
        features = np.array([[age, sex_val[0], cp_val[0], trestbps, chol, fbs_val[0], 
                             restecg_val[0], thalach, exang_val[0], oldpeak, 
                             slope_val[0], ca, thal_val[0]]])
        
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1] * 100
        result_text = "High Risk" if prediction == 1 else "Low Risk"

       # DATABASE ---
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history_data = (
            timestamp, age, sex_val[1], cp_val[1], trestbps, chol, fbs_val[1],
            restecg_val[1], thalach, exang_val[1], oldpeak, slope_val[1], 
            ca, thal_val[1], result_text, round(probability, 2)
        )
        add_history(history_data)

        st.write("---")
        if prediction == 1:
            st.markdown(f"""<div style="background-color: #ffcccc; padding: 20px; border-radius: 10px; border-left: 10px solid #c0392b;">
                <h2 style="color: #c0392b;">⚠️ High Risk Detected</h2>
                <p>Probability: <b>{probability:.1f}%</b>. Consult a cardiologist.</p></div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div style="background-color: #d4edda; padding: 20px; border-radius: 10px; border-left: 10px solid #28a745;">
                <h2 style="color: #155724;">✅ Low Risk Detected</h2>
                <p>Confidence: <b>{100-probability:.1f}%</b>. Healthy metrics observed.</p></div>""", unsafe_allow_html=True)

with tab2:
    st.markdown("<h1>📜 Patient Diagnostic History</h1>", unsafe_allow_html=True)
    
    conn = sqlite3.connect('heart_history.db')
    try:
        df_history = pd.read_sql_query("SELECT * FROM medical_history ORDER BY id DESC", conn)
        
        if not df_history.empty:
            st.dataframe(df_history, use_container_width=True)
            
            csv = df_history.to_csv(index=False).encode('utf-8')
            st.download_button("Download History as CSV", csv, "heart_history.csv", "text/csv")
            
            st.write("---")
            col_stat1, col_stat2 = st.columns(2)
            col_stat1.metric("Total Assessments", len(df_history))
            high_risk_count = len(df_history[df_history['prediction'] == "High Risk"])
            col_stat2.metric("High Risk Cases", high_risk_count)
        else:
            st.info("No history found. Perform a prediction to see data here.")
    except:
        st.error("Could not load history.")
    finally:
        conn.close()

st.markdown("<br><br><p style='text-align: center; font-size: 12px; color: #999;'>Advancing cardiovascular wellness through technology. ❤️</p>", unsafe_allow_html=True)