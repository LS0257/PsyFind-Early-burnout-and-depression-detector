import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

# Page configuration
st.set_page_config(
    page_title="Burnout & Depression Detection",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #ffebee;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #f44336;
    }
    .risk-moderate {
        background-color: #fff3e0;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff9800;
    }
    .risk-low {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🧠 Burnout & Depression Risk Assessment</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("""
    This AI-powered system uses Machine Learning and NLP to assess 
    burnout and depression risk based on multiple factors.
    
    **Technology Stack:**
    - Machine Learning (Random Forest)
    - Natural Language Processing
    - RAG (Retrieval-Augmented Generation)
    """)
    
    st.header("⚠️ Disclaimer")
    st.warning("""
    This tool is for screening purposes only and does not replace 
    professional medical advice. If you're in crisis, please contact 
    a mental health professional immediately.
    
    **Crisis Resources:**
    - National Suicide Prevention: 988
    - Crisis Text Line: Text HOME to 741741
    """)

# Load models
@st.cache_resource
def load_models():
    try:
        model = joblib.load('models/burnout_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        target_encoder = joblib.load('models/target_encoder.pkl')
        label_encoders = joblib.load('models/label_encoders.pkl')
        feature_columns = joblib.load('models/feature_columns.pkl')
        
        with open('models/recommendations_db.json', 'r') as f:
            recommendations_db = json.load(f)
        
        return model, scaler, target_encoder, label_encoders, feature_columns, recommendations_db
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please run the training notebook first to generate model files.")
        return None, None, None, None, None, None

model, scaler, target_encoder, label_encoders, feature_columns, recommendations_db = load_models()

# Text preprocessing function
def preprocess_text(text):
    if pd.isna(text) or text == "":
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

# Prediction function
def predict_risk(user_data):
    if model is None:
        return None
    
    negative_words = ['anxious', 'depressed', 'stress', 'panic', 'worry', 'fear', 'pain', 
                      'exhausted', 'tired', 'overwhelmed', 'sad', 'hopeless', 'lonely']
    
    categorical_features = ['Gender', 'Education_Level', 'Employment_Status', 'Substance_Use']
    
    # Preprocess text
    user_data['Processed_Text'] = preprocess_text(user_data['Text'])
    user_data['Text_Length'] = len(user_data['Text'])
    user_data['Word_Count'] = len(user_data['Text'].split())
    user_data['Negative_Word_Count'] = sum(
        1 for word in negative_words if word in user_data['Processed_Text'].lower()
    )
    
    # Encode categorical features
    for col in categorical_features:
        user_data[col + '_Encoded'] = label_encoders[col].transform([user_data[col]])[0]
    
    # Prepare features
    features = [user_data[col] for col in feature_columns]
    features_scaled = scaler.transform([features])
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    risk_category = target_encoder.inverse_transform([prediction])[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    # Get recommendations
    recommendations = recommendations_db[risk_category]
    
    return {
        'risk_category': risk_category,
        'confidence': max(probabilities) * 100,
        'probabilities': dict(zip(target_encoder.classes_, probabilities * 100)),
        'recommendations': recommendations
    }

# Main form
if model is not None:
    st.header("📋 Please fill in your information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Gender", ["Male", "Female", "Non-Binary", "Other"])
        education = st.selectbox("Education Level", 
                                ["High School", "Bachelor's", "Master's", "PhD", "Other"])
        employment = st.selectbox("Employment Status", 
                                 ["Employed", "Unemployed", "Retired", "Student"])
    
    with col2:
        sleep_hours = st.slider("Average Sleep Hours per Night", 0.0, 12.0, 7.0, 0.1)
        anxiety_score = st.slider("Anxiety Score (0-20)", 0, 20, 5)
        depression_score = st.slider("Depression Score (0-20)", 0, 20, 5)
        stress_level = st.slider("Stress Level (0-10)", 0, 10, 5)
    
    with col3:
        family_history = st.selectbox("Family History of Mental Illness", [0, 1], 
                                     format_func=lambda x: "Yes" if x == 1 else "No")
        chronic_illness = st.selectbox("Chronic Illnesses", [0, 1], 
                                      format_func=lambda x: "Yes" if x == 1 else "No")
        substance_use = st.selectbox("Substance Use", 
                                     ["none", "Occasional", "Frequent"])
        financial_stress = st.slider("Financial Stress (0-10)", 0, 10, 5)
    
    col4, col5 = st.columns(2)
    with col4:
        work_stress = st.slider("Work Stress (0-10)", 0, 10, 5)
        self_esteem = st.slider("Self-Esteem Score (0-10)", 0, 10, 5)
    with col5:
        loneliness = st.slider("Loneliness Score (0-10)", 0, 10, 5)
    
    st.markdown("---")
    st.subheader("💭 How are you feeling?")
    user_text = st.text_area(
        "Please describe your current mental state, feelings, or concerns (the more detail, the better):",
        height=150,
        placeholder="Example: I've been feeling overwhelmed with work lately, having trouble sleeping, and experiencing frequent anxiety..."
    )
    
    st.markdown("---")
    
    if st.button("🔍 Assess My Risk", type="primary", use_container_width=True):
        if user_text.strip() == "":
            st.warning("Please describe how you're feeling in the text box above.")
        else:
            with st.spinner("Analyzing your data..."):
                user_data = {
                    'Age': age,
                    'Gender': gender,
                    'Education_Level': education,
                    'Employment_Status': employment,
                    'Sleep_Hours': sleep_hours,
                    'Anxiety_Score': anxiety_score,
                    'Depression_Score': depression_score,
                    'Stress_Level': stress_level,
                    'Family_History_Mental_Illness': family_history,
                    'Chronic_Illnesses': chronic_illness,
                    'Substance_Use': substance_use,
                    'Financial_Stress': financial_stress,
                    'Work_Stress': work_stress,
                    'Self_Esteem_Score': self_esteem,
                    'Loneliness_Score': loneliness,
                    'Text': user_text
                }
                
                result = predict_risk(user_data)
                
                st.markdown("---")
                st.header("📊 Assessment Results")
                
                # Display risk category with appropriate styling
                risk_cat = result['risk_category']
                if risk_cat == "High Risk":
                    st.markdown(f'<div class="risk-high"><h2>⚠️ Risk Level: {risk_cat}</h2><p>Confidence: {result["confidence"]:.1f}%</p></div>', 
                               unsafe_allow_html=True)
                elif risk_cat == "Moderate Risk":
                    st.markdown(f'<div class="risk-moderate"><h2>⚡ Risk Level: {risk_cat}</h2><p>Confidence: {result["confidence"]:.1f}%</p></div>', 
                               unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="risk-low"><h2>✅ Risk Level: {risk_cat}</h2><p>Confidence: {result["confidence"]:.1f}%</p></div>', 
                               unsafe_allow_html=True)
                
                # Probability breakdown
                st.subheader("Risk Probability Breakdown")
                prob_df = pd.DataFrame({
                    'Risk Category': result['probabilities'].keys(),
                    'Probability (%)': [f"{v:.1f}%" for v in result['probabilities'].values()]
                })
                st.dataframe(prob_df, use_container_width=True)
                
                # Recommendations
                st.markdown("---")
                st.header("💡 Personalized Recommendations")
                
                for category, items in result['recommendations'].items():
                    with st.expander(f"📌 {category.replace('_', ' ').title()}", expanded=True):
                        for i, item in enumerate(items, 1):
                            st.markdown(f"**{i}.** {item}")
                
                # Additional resources
                st.markdown("---")
                st.info("""
                ### 🆘 Need Immediate Help?
                
                If you're experiencing a mental health crisis:
                - **Call 988** - National Suicide Prevention Lifeline
                - **Text HOME to 741741** - Crisis Text Line
                - **Call 911** - If you're in immediate danger
                - **Visit your nearest emergency room**
                
                Remember: Seeking help is a sign of strength, not weakness. 💙
                """)

else:
    st.error("Models not loaded. Please run the training notebook first.")
    st.info("Run the Jupyter notebook to train the model and generate required files.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Developed for Final Year Project | Burnout & Depression Detection System</p>
    <p>⚠️ This is a screening tool, not a diagnostic tool. Please consult healthcare professionals.</p>
</div>
""", unsafe_allow_html=True)
