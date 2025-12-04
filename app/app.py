from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import random
import numpy as np 

app = Flask(__name__)
app.secret_key = 'psyfind_secret_key_2025'

# --- 1. SETUP & CONFIGURATION ---
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass
# --- 2. LOAD MODELS ---
model = scaler = le_gender = le_education = le_employment = le_substance = feature_columns = None
try:
    model = joblib.load(os.path.join(MODELS_DIR, "burnout_depression_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    le_gender = joblib.load(os.path.join(MODELS_DIR, "gender_encoder.pkl"))
    le_education = joblib.load(os.path.join(MODELS_DIR, "education_encoder.pkl"))
    le_employment = joblib.load(os.path.join(MODELS_DIR, "employment_encoder.pkl"))
    le_substance = joblib.load(os.path.join(MODELS_DIR, "substance_encoder.pkl"))
    feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
    print("✓ All models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
# --- 3. KNOWLEDGE BASE (RAG) ---
# NOTE: Keeping the RAG DB structure the same as it is the source for LLM advice.
recommendations_db_enhanced = {
    'Depression': {
        'Professional': [
            "Consult a licensed mental health therapist (e.g., Cognitive Behavioral Therapy, Dialectical Behavior Therapy, Interpersonal Therapy).",
            "Book an appointment with a psychiatrist to discuss pharmacological treatment options (medication).",
            "Utilize digital mental health tools and teletherapy for accessible care.",
            "Undergo a comprehensive medical check-up to rule out physical causes (e.g., thyroid issues, vitamin deficiencies)."
        ],
        'Lifestyle': [
            "Aim for 7-9 hours of consistent, quality sleep (establish a strict, consistent bedtime and 'wind-down' routine).",
            "Engage in physical activity for at least 30 minutes, 5 days a week (prioritize movement you genuinely enjoy).",
            "Adopt a Mediterranean-style or whole-foods diet rich in Omega-3s, focusing on unprocessed grains, vegetables, and lean protein.",
            "Limit alcohol and strictly avoid recreational drugs (these interfere with mood regulation and sleep)."
        ],
        'Coping': [
            "Practice mindfulness and meditation for 10-15 minutes daily (use guided apps if needed).",
            "Keep a mood journal to track patterns, emotional triggers, and positive events (not just negative ones).",
            "Connect with friends and family regularly for social support and reduce feelings of isolation.",
            "Practice Gratitude daily: write down 3 things you are thankful for, no matter how small, to reframe perspective."
        ]
    },
    'Burnout': {
        'Work_Boundaries': [
            "Set clear boundaries: turn off ALL work notifications after hours and define a strict 'End of Day' ritual.",
            "Discuss workload, resources, or flexible hours with your supervisor/HR to create a sustainable structure.",
            "Delegate tasks when possible and learn to politely decline new, non-essential commitments (i.e., establish a 'No List').",
            "Adopt a 'mono-tasking' mindset; avoid context-switching to increase deep work and efficiency."
        ],
        'Energy_Management': [
            "Take micro-breaks (5-10 minutes every hour) for simple stretches or movement away from the screen (e.g., the Pomodoro Technique).",
            "Ensure adequate sleep and maintain a healthy, regular meal schedule to stabilize blood sugar and energy.",
            "Take vacation time or a personal day to completely disconnect and recharge (no email checking!).",
            "Integrate outdoor time (green exercise) into your day, even if it's just a short walk."
        ],
        'Stress_Relief': [
            "Engage in hobbies and activities you enjoy outside of work to rebuild a sense of competence and joy.",
            "Practice stress management techniques like deep-breathing (Box Breathing) or Progressive Muscle Relaxation (PMR).",
            "Seek out or join a support group focused on occupational stress or workplace issues.",
            "Use music or sound therapy (e.g., binaural beats, calming playlists) to induce relaxation during breaks."
        ]
    },
    'Both': {
        'Immediate_Action': [
            "Seek immediate professional help for a comprehensive assessment by a specialist (combined therapy/medication is often required).",
            "Inform a close family member or friend of your condition for accountability and support (an emergency contact).",
            "Create a crisis plan detailing who to call and what steps to take if thoughts of self-harm arise (e.g., a 24/7 hotline).",
            "If symptoms are debilitating, evaluate your work situation and consider temporary medical leave (e.g., FMLA)."
        ],
        'Integrated_Care': [
            "Prioritize a holistic treatment plan that addresses both emotional distress (Depression) and structural demands (Burnout).",
            "Practice self-compassion by treating yourself with the same kindness you would offer a friend in distress.",
            "Work with your therapist/psychiatrist and your HR department (if applicable) for a gradual return-to-work plan.",
            "Actively challenge cognitive distortions (e.g., all-or-nothing thinking, blaming) common in both states."
        ],
        'Foundational_Self_Care': [
            "Establish a simple, structured daily routine focusing on basic needs (meals, hydration, light movement, hygiene).",
            "Build a strong, reliable support system (professionals, friends, family) and accept help when offered.",
            "Reduce exposure to high-stress media, including constant news consumption and draining social media feeds.",
            "Focus on one small, achievable task each day to rebuild a sense of mastery and momentum."
        ]
    },
    'Healthy': {
        'Maintenance': [
            "Continue your current healthy lifestyle and habits, treating them as non-negotiable commitments.",
            "Schedule regular physical and mental health check-ups (e.g., check-in with your primary care provider annually).",
            "Maintain a growth mindset by regularly learning new skills or engaging in creative outlets to challenge the mind.",
            "Document your 'Early Warning Signs' list (e.g., irritability, skipping workouts, poor sleep) and review it monthly."
        ],
        'Prevention': [
            "Incorporate a new stress-busting activity (e.g., learning a skill, pottery, music) to broaden your coping toolkit.",
            "Regularly review your work-life balance using a weekly audit (e.g., how many hours were purely dedicated to relaxation/self-care).",
            "Practice proactive digital detoxes—schedule specific blocks of time away from screens and social media.",
            "Engage in volunteering or acts of kindness to boost a sense of purpose and connection."
        ],
        'Social_Wellbeing': [
            "Stay socially connected with loved ones and community (e.g., join a club, take a class).",
            "Be aware of early warning signs (fatigue, irritability) and address them promptly using your coping skills.",
            "Nurture your core relationships by scheduling quality, distraction-free time with family and friends.",
            "Periodically re-evaluate your life goals and values to ensure your actions align with your personal purpose."
        ]
    }
}
# --- 4. HELPER FUNCTIONS ---
def clean_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    return ' '.join([w for w in tokens if w not in stop_words and len(w) > 2])
def retrieve_context_enhanced(prediction, user_data):
    """Retrieves two structured, diverse context points for the LLM Advisor."""
    lookup_key = 'Both' if 'Both' in prediction else prediction
    topic_db = recommendations_db_enhanced.get(lookup_key, recommendations_db_enhanced['Healthy'])
    key_areas = list(topic_db.keys())
    # Strategy to ensure diversity and priority
    if lookup_key != 'Healthy':
        # Prioritize 'Immediate_Action' if 'Both' is predicted
        area_1 = 'Immediate_Action' if lookup_key == 'Both' and 'Immediate_Action' in key_areas else random.choice(key_areas)
        remaining = [a for a in key_areas if a != area_1]
        area_2 = random.choice(remaining) if remaining else area_1
    else: # Healthy - prioritize Maintenance
        area_1 = 'Maintenance'
        remaining = [a for a in key_areas if a != 'Maintenance']
        area_2 = random.choice(remaining) if remaining else 'Maintenance'
    return {
        area_1: random.sample(topic_db[area_1], 1)[0],
        area_2: random.sample(topic_db[area_2], 1)[0]
    }
def generate_llm_response(prediction, user_data):
    """Generates a cohesive, empathetic advisor message using RAG context and personalization."""
    context = retrieve_context_enhanced(prediction, user_data)
    # 1. Opening Statement (Empathetic)
    if 'Both' in prediction:
        # Note: Using Markdown bolding here, removed on the front-end rendering
        status_msg = f"Thank you for sharing what you are going through. From the way you described things and your high stress and emotional scores, it sounds like you are carrying more than anyone should have to handle alone. Feeling drained, low, or overwhelmed makes sense when both **Burnout and Depression** are weighing on you."
    elif prediction == 'Burnout':
        status_msg = f"I appreciate your honesty. With your work stress at {user_data['Work_Stress']}/10 and the amount of pressure you are under, it is completely understandable that you are feeling stretched thin. Nothing about your reactions is 'wrong' — they are signals that you have been pushing past your limits for too long."
    elif prediction == 'Depression':
        status_msg = f"Thank you for speaking so openly. The heaviness you described — especially combined with your self-esteem score of {user_data['Self_Esteem_Score']}/10 — tells me that you have been trying to get through each day with a weight that is hard for anyone to carry. Feeling disconnected or numb doesn’t mean you are failing. It means you are hurting."
    else:  # Healthy
        status_msg = "Your responses show that you are in a stable place emotionally right now, which is really good to see. Even so, staying grounded and protecting your mental space is something worth doing intentionally."
    # 2. Actionable Advice (RAG + Personalization)
    advice_paragraphs = []
    advice_paragraphs.append("Here are a few small steps that could help you:")
    # RAG Context
    for area, tip in context.items():
        area_clean = area.replace('_', ' ').title()
        # Note: Bolding the area for emphasis in the output
        advice_paragraphs.append(f"• **{area_clean}:** {tip}")
    # Data-Driven Personalization
    if prediction not in ['Healthy']:
        if user_data['Sleep_Hours'] < 6.0:
            advice_paragraphs.append(f"• **Personalized Sleep Priority:** You are running on very little sleep ({user_data['Sleep_Hours']} hours). Tonight, try choosing one small thing that signals your mind it is time to unwind — putting your phone away earlier, dimming the lights, or making a warm drink. Small cues can shift your whole night.")
        if user_data['Loneliness_Score'] >= 8:
            advice_paragraphs.append(f"• **Connection Priority:** Your loneliness score ({user_data['Loneliness_Score']}/10) tells me you have been feeling disconnected. You do not need a long conversation — even sending a short check-in to someone you trust can make you feel less alone.")
        if user_data['Work_Stress'] >= 8 and prediction in ['Burnout', 'Both']:
            advice_paragraphs.append("• **Boundary Priority:** The pressure you are facing at work seems intense. Tomorrow, choose one task you can either postpone, simplify, or say 'no' to. Even a small boundary can help you regain a sense of control.")
    if prediction == 'Healthy':
        advice_paragraphs.append(
            "• **Maintenance Priority:** Since you are doing well, consider choosing one weekly habit — a walk, journaling, a quiet hour — that keeps you grounded moving forward."
        )
    # 3. Soft Closing Message
    closing = "You do not need to fix everything at once. Just taking one gentle step today is enough. What you are feeling is valid, and you deserve care, patience, and support while moving through it."
    return {
        "opening": status_msg,
        "advice": advice_paragraphs,
        "closing": closing
    }
def predict_user_status(user_data):
    if model is None or scaler is None or le_gender is None:
        raise Exception("Model artifacts not loaded. Prediction aborted.")
    text_input = user_data.get('Text', '')
    cleaned_text = clean_text(text_input)
    def safe_transform(encoder, val):
        try: return encoder.transform([val])[0]
        except: return 0 
    features = pd.DataFrame([{
        'Age': user_data['Age'],
        'Gender_Encoded': safe_transform(le_gender, user_data['Gender']),
        'Education_Encoded': safe_transform(le_education, user_data['Education_Level']),
        'Employment_Encoded': safe_transform(le_employment, user_data['Employment_Status']),
        'Sleep_Hours': user_data['Sleep_Hours'],
        'Anxiety_Score': user_data['Anxiety_Score'], # Note: Model expects 0-20, but we pass 0-10 from frontend. Max 10.
        'Stress_Level': user_data['Stress_Level'],
        'Family_History_Mental_Illness': user_data['Family_History_Mental_Illness'],
        'Chronic_Illnesses': user_data['Chronic_Illnesses'],
        'Substance_Encoded': safe_transform(le_substance, user_data['Substance_Use']),
        'Financial_Stress': user_data['Financial_Stress'],
        'Work_Stress': user_data['Work_Stress'],
        'Self_Esteem_Score': user_data['Self_Esteem_Score'],
        'Loneliness_Score': user_data['Loneliness_Score'],
        'Text_Length': len(cleaned_text.split()),
        'Char_Count': len(text_input)
    }])
    features_scaled = scaler.transform(features)
    pred_label = model.predict(features_scaled)[0]
    probs = model.predict_proba(features_scaled)[0]
    if pred_label == 'Both':
        pred_label = "Both (Burnout and Depression)"
    return pred_label, dict(zip(model.classes_, probs))
# --- 5. ROUTES ---
@app.route('/')
def home():
    return render_template('template.html')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        user_data = {
            'Age': int(data.get('age', 30)),
            'Gender': data.get('gender', 'Other'),
            'Education_Level': data.get('education', "Bachelor's"),
            'Employment_Status': data.get('employment', 'Employed'),
            'Sleep_Hours': float(data.get('sleep_hours', 7.0)),
            'Anxiety_Score': int(data.get('anxiety_score', 0)),
            'Stress_Level': int(data.get('stress_level', 0)),
            'Family_History_Mental_Illness': int(data.get('family_history', 0)),
            'Chronic_Illnesses': int(data.get('chronic_illness', 0)),
            'Substance_Use': data.get('substance_use', 'none'),
            'Financial_Stress': int(data.get('financial_stress', 0)),
            'Work_Stress': int(data.get('work_stress', 0)),
            'Self_Esteem_Score': int(data.get('self_esteem', 5)),
            'Loneliness_Score': int(data.get('loneliness_score', 0)),
            'Text': data.get('text_description', '')
        }
        prediction, probabilities = predict_user_status(user_data)
        advisor_response = generate_llm_response(prediction, user_data)
        return jsonify({
            'prediction': prediction,
            'probabilities': {k: float(v) for k, v in probabilities.items()},
            'advisor_response': advisor_response
        })
    except Exception as e:
        print(f"Prediction Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': "An internal error occurred during analysis. Check server logs."}), 500
if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True, port=5000)