"""
Emergency deployment script - creates a single file that can be easily deployed
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🏥",
    layout="wide"
)

@st.cache_data
def load_data():
    """Generate synthetic diabetes data for demo"""
    np.random.seed(42)
    n = 1000
    
    data = pd.DataFrame({
        'Pregnancies': np.random.poisson(3, n),
        'Glucose': np.random.normal(120, 30, n),
        'BloodPressure': np.random.normal(70, 12, n),
        'SkinThickness': np.random.normal(25, 8, n),
        'Insulin': np.random.exponential(100, n),
        'BMI': np.random.normal(28, 6, n),
        'DiabetesPedigreeFunction': np.random.exponential(0.5, n),
        'Age': np.random.normal(35, 12, n)
    })
    
    # Create realistic outcome
    risk = (
        (data['Glucose'] - 100) * 0.02 +
        (data['BMI'] - 25) * 0.05 +
        (data['Age'] - 30) * 0.01 +
        data['DiabetesPedigreeFunction'] * 0.3 +
        np.random.normal(0, 0.5, n)
    )
    data['Outcome'] = (risk > 0.5).astype(int)
    return data

@st.cache_resource
def train_models():
    """Train all models"""
    data = load_data()
    
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    X = data[features]
    y = data['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }
    
    trained = {}
    for name, model in models.items():
        if name in ['Logistic Regression', 'SVM']:
            model.fit(X_train_scaled, y_train)
        else:
            model.fit(X_train, y_train)
        trained[name] = model
    
    return trained, scaler, features

def main():
    st.title("🏥 Diabetes Risk Prediction System")
    st.markdown("AI-powered diabetes risk assessment using multiple machine learning models")
    
    st.warning("⚠️ Medical Disclaimer: This tool is for educational purposes only. Consult healthcare providers for medical decisions.")
    
    models, scaler, features = train_models()
    
    st.sidebar.header("Patient Information")
    
    inputs = {
        'Pregnancies': st.sidebar.number_input("Pregnancies", 0, 17, 1),
        'Glucose': st.sidebar.number_input("Glucose (mg/dL)", 0, 200, 120),
        'BloodPressure': st.sidebar.number_input("Blood Pressure", 0, 122, 70),
        'SkinThickness': st.sidebar.number_input("Skin Thickness", 0, 99, 25),
        'Insulin': st.sidebar.number_input("Insulin", 0, 846, 80),
        'BMI': st.sidebar.number_input("BMI", 0.0, 67.1, 25.0),
        'DiabetesPedigreeFunction': st.sidebar.number_input("Pedigree Function", 0.0, 2.5, 0.5),
        'Age': st.sidebar.number_input("Age", 21, 81, 30)
    }
    
    if st.sidebar.button("Predict Risk", type="primary"):
        input_df = pd.DataFrame([inputs])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Predictions")
            predictions = {}
            
            for name, model in models.items():
                if name in ['Logistic Regression', 'SVM']:
                    input_scaled = scaler.transform(input_df[features])
                    prob = model.predict_proba(input_scaled)[0][1]
                else:
                    prob = model.predict_proba(input_df[features])[0][1]
                
                predictions[name] = prob
                
                if prob < 0.3:
                    risk_level = "Low Risk"
                    color = "green"
                elif prob < 0.7:
                    risk_level = "Moderate Risk"
                    color = "orange"
                else:
                    risk_level = "High Risk"
                    color = "red"
                
                st.metric(name, f"{prob:.1%}", risk_level)
        
        with col2:
            st.subheader("Risk Visualization")
            
            fig = go.Figure()
            model_names = list(predictions.keys())
            probs = list(predictions.values())
            colors = ['green' if p < 0.3 else 'orange' if p < 0.7 else 'red' for p in probs]
            
            fig.add_trace(go.Bar(
                x=model_names,
                y=probs,
                marker_color=colors,
                text=[f"{p:.1%}" for p in probs],
                textposition='auto'
            ))
            
            fig.update_layout(
                title="Diabetes Risk by Model",
                xaxis_title="Models",
                yaxis_title="Risk Probability",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        avg_prob = np.mean(list(predictions.values()))
        st.subheader("Overall Assessment")
        st.write(f"Average Risk Score: {avg_prob:.1%}")
        
        st.subheader("Recommendations")
        if avg_prob > 0.5:
            st.write("🚨 Consult healthcare provider")
            st.write("📊 Consider glucose tolerance test")
        
        if inputs['BMI'] > 25:
            st.write("🏃‍♂️ Focus on weight management")
        
        if inputs['Glucose'] > 140:
            st.write("🍎 Monitor blood sugar levels")
        
        st.write("💪 Maintain regular exercise")
        st.write("🥗 Follow balanced diet")

if __name__ == "__main__":
    main()