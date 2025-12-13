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
# --- ENHANCED KNOWLEDGE BASE (CONTEXTUALIZED FOR INDIAN USER BASE) ---
recommendations_db_enhanced = {
    'Depression': {
        'Get_Help_Now': [
            "Talk to a licensed mental health professional, like a counsellor or therapist. Ask about options like **'Talk Therapy'** (CBT, DBT).",
            "See a psychiatrist to discuss if medicine could help you feel better.",
            "Try **online counseling or teletherapy**—it's often easier and more private than in-person visits.",
            "Ask your doctor for a **full physical check-up** to ensure no physical issues (like low vitamins or thyroid problems) are making your mood worse."
        ],
        'Daily_Habits': [
            "Aim for 7 to 9 hours of **good sleep** every night. Try to go to bed and wake up at the same time, even on weekends.",
            "Move your body for at least 30 minutes, 5 days a week. It doesn't have to be a hard workout—a **simple walk or Yoga** is great.",
            "Eat healthy foods like vegetables, beans, and whole grains. Focus on **fresh, unprocessed food** to keep your energy steady.",
            "Limit alcohol and **do not use recreational drugs**—they stop your brain from managing your mood correctly.",
            "Get outside and see the **sunlight** in the morning; it helps your body clock stay balanced."
        ],
        'Mind_Tools': [
            "Practice **mindfulness or meditation** for 10-15 minutes daily to calm your mind.",
            "Start a **simple mood journal** to track when you feel bad, but also write down at least one *good thing* that happened each day.",
            "Make time to call or meet up with friends and family for **social support**.",
            "Practice **Gratitude** daily: write down 3 things you are thankful for, no matter how small, to change your view on life."
        ]
    },
    'Burnout': {
        'Work_Limits': [
            "Set firm digital boundaries: Define a non-negotiable **workday end time** and commit to **turning off all work-related apps and notifications** thereafter.",
            "Talk to your manager or HR about **reducing your workload** or changing your hours so you can cope.",
            "Ask others to do tasks you don't need to do (delegate). **Learn to say 'No' politely** to new, non-essential work.",
            "Focus on **one task at a time** (mono-tasking); switching between tasks constantly wastes your energy."
        ],
        'Energy_Refresh': [
            "Take **short breaks** (5-10 minutes) every hour to stand up, stretch, and get away from your screen.",
            "Make sure you eat **regular, healthy meals** throughout the day to keep your energy levels stable.",
            "Use your **vacation days or take a personal day** to fully disconnect and recharge (no checking email!).",
            "Get some **fresh air and green time** every day, even just for a quick walk in the park."
        ],
        'Stress_Release': [
            "Spend time on **hobbies and activities** you genuinely enjoy outside of work to feel successful and happy again.",
            "Use stress-relief methods like **deep-breathing (Box Breathing)** or relaxing your muscles one by one (PMR).",
            "Find a **support group** or connect with people who share your job stress.",
            "Use **calming music or nature sounds** to help you relax during your breaks."
        ]
    },
    'Both': {
        'Immediate_Action': [
            "Get **professional help right away**. You need an expert to treat both your emotional distress and your overwhelming stress.",
            "Tell a trusted family member or friend about your condition. They can be your **emergency contact and support buddy**.",
            "If you have thoughts of self-harm, immediately use the **24/7 helplines** listed on this screen.",
            "If your symptoms are stopping you from functioning, discuss taking **temporary time off work or study** for medical recovery."
        ],
        'Integrated_Care': [
            "Focus on a **full treatment plan** that handles both your low mood (Depression) and your exhaustion (Burnout).",
            "Be **kind to yourself**. Treat yourself with the same care and patience you would give a friend who is struggling.",
            "Work with your mental health expert and workplace to create a plan for a **slow and gentle return to work**.",
            "Actively challenge **negative thoughts** like 'I always fail' or 'It's all my fault'—these thoughts are often incorrect."
        ],
        'Basic_Self_Care': [
            "Create a **simple, structured daily plan** focused on your basic needs: eating, drinking water, gentle movement, and getting ready.",
            "Build a strong and reliable **support system** (experts, friends, family) and **accept help** when it is offered.",
            "**Reduce time spent on stressful news** or draining social media feeds.",
            "Focus on completing **one small, easy task** each day to build momentum and feel a sense of achievement."
        ]
    },
    'Healthy': {
        'Stay_Healthy': [
            "Keep doing your healthy habits—they are your **non-negotiable** commitments to yourself.",
            "Schedule **regular physical and mental health check-ups** with your doctor every year.",
            "Keep challenging your mind by **learning new skills** or working on creative projects.",
            "Write down your **'Early Warning Signs'** list (e.g., getting easily annoyed, skipping workouts) and check it monthly."
        ],
        'Guard_Your_Health': [
            "Add a **new way to handle stress** to your routine (e.g., learning an art form, playing music, new sport).",
            "Use a weekly check to **review your work-life balance** (e.g., how many hours did you spend purely on fun and relaxation).",
            "Plan for **regular screen-free time** away from phones and social media (Digital Detox).",
            "Do **volunteer work or acts of kindness** to boost your sense of purpose and connection."
        ],
        'Strong_Connections': [
            "Stay **socially active** with loved ones and community (e.g., join a club or class).",
            "If you notice early signs of fatigue or stress, **address them right away** using your usual coping skills.",
            "Make time for quality, **distraction-free time** with your family and friends.",
            "Every few months, **think about your life goals and values** to make sure your daily actions match what is important to you."
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