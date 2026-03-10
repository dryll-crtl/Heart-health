import streamlit as st
import pickle
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime
import plotly.graph_objects as go

#DATABASE FUNCTIONS
def init_db():
    conn = sqlite3.connect('heart_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT,
                  age REAL, bp REAL, chol REAL, bmi REAL, glucose REAL, 
                  gender TEXT, result TEXT, probability REAL)''')
    conn.commit()
    conn.close()

def save_prediction(age, bp, chol, bmi, glucose, gender, result, prob):
    conn = sqlite3.connect('heart_history.db')
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO predictions (timestamp, age, bp, chol, bmi, glucose, gender, result, probability) VALUES (?,?,?,?,?,?,?,?,?)",
              (now, age, bp, chol, bmi, glucose, gender, result, prob))
    conn.commit()
    conn.close()

st.set_page_config(page_title="Heart Health", layout="wide", page_icon="❤️")
init_db()

try:
    model = pickle.load(open("heart_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except FileNotFoundError:
    st.error("⚠️ Files missing! Please run 'Train_model.py' first to generate heart_model.pkl and scaler.pkl.")

st.title("❤️ Heart Disease Diagnostic Dashboard")
st.markdown("Professional-grade AI prediction based on clinical metrics.")
st.markdown("---")

#INPUTS & PREDICTION 
col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.subheader("📋 Patient Metrics")
    
    r1_c1, r1_c2 = st.columns(2)
    age = r1_c1.number_input("Age", 1, 120, 30)
    gender_label = r1_c2.selectbox("Gender", ["Male", "Female"])
    gender_val = 1 if gender_label == "Male" else 0

    r2_c1, r2_c2 = st.columns(2)
    bp = r2_c1.number_input("Blood Pressure (mmHg)", 50, 250, 120)
    chol = r2_c2.number_input("Cholesterol (mg/dL)", 100, 600, 200)

    r3_c1, r3_c2 = st.columns(2)
    bmi = r3_c1.number_input("BMI Index", 10.0, 60.0, 25.0)
    glucose = r3_c2.number_input("Glucose Level", 50, 300, 100)

    predict_btn = st.button("Analyze Results", use_container_width=True, type="primary")

with col2:
    if predict_btn:
    
        raw_features = np.array([[age, bp, chol, bmi, glucose, gender_val]])
        scaled_features = scaler.transform(raw_features)
        
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1] * 100 
        
        result_text = "High Risk" if prediction == 1 else "Low Risk"
        result_color = "#ff4b4b" if prediction == 1 else "#2ebd59"

        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probability,
            title = {'text': "Disease Probability (%)", 'font': {'size': 20}},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': result_color},
                'steps': [
                    {'range': [0, 40], 'color': "#e8f5e9"},
                    {'range': [40, 70], 'color': "#fff3e0"},
                    {'range': [70, 100], 'color': "#ffebee"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50}}))
        
        fig.update_layout(height=350, margin=dict(t=50, b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
            <div style="background-color:{result_color}; padding:25px; border-radius:15px; text-align:center; color:white;">
                <h1 style="margin:0; font-size:40px;">Result: {result_text}</h1>
                <p style="margin:0; font-size:20px; opacity:0.9;">Confidence Level: {probability:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        save_prediction(age, bp, chol, bmi, glucose, gender_label, result_text, probability)

#ANALYTICS & HISTORY SECTION 
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.subheader("📊 Analytics & History")

tab1, tab2, tab3 = st.tabs(["🕒 History Log", "📈 Health Trends", "📋 Recommendations"])

conn = sqlite3.connect('heart_history.db')
all_history_df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
conn.close()

with tab1:
    if not all_history_df.empty:
        st.dataframe(all_history_df[['timestamp', 'age', 'gender', 'result', 'probability']].head(10), use_container_width=True)
        
        csv = all_history_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Full History as CSV", data=csv, file_name='heart_data_history.csv', mime='text/csv')
        
        if st.button("🗑️ Clear All Records"):
            conn = sqlite3.connect('heart_history.db')
            conn.cursor().execute("DELETE FROM predictions")
            conn.commit()
            conn.close()
            st.success("Database cleared.")
            st.rerun()
    else:
        st.info("No prediction history found yet.")

with tab2:
    if len(all_history_df) > 1:
        st.markdown("##### Probability Trend Analysis")
        trend_df = all_history_df.sort_values(by='timestamp')
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=trend_df['timestamp'], 
            y=trend_df['probability'],
            mode='lines+markers',
            line=dict(color='#ff4b4b', width=3),
            marker=dict(size=8, color='white', line=dict(width=2, color='#ff4b4b'))
        ))
        fig_trend.update_layout(xaxis_title="Checkup Date", yaxis_title="Risk Probability %", yaxis=dict(range=[0,105]))
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.warning("Please make at least two predictions to view trend data.")

with tab3:
    if not all_history_df.empty:
        latest_res = all_history_df.iloc[0]['result']
        latest_prob = all_history_df.iloc[0]['probability']
        
        st.markdown(f"### Medical Guidance for {latest_res}")
        if latest_res == "High Risk":
            st.error("🚩 **High Risk Detected:** It is highly recommended to book an appointment with a cardiologist. Reduce salt intake, avoid smoking, and monitor your blood pressure daily.")
        else:
            st.success("✅ **Low Risk Detected:** Your current metrics are stable. Continue a balanced diet, maintain 150 minutes of weekly exercise, and schedule an annual checkup.")
    else:
        st.info("Recommendations will appear after your first analysis.")

