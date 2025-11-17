import pandas as pd
import joblib
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import chromadb

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

# Load models
print("\nLoading trained models...")
try:
    # Load saved models
    model = joblib.load(r"C:\Users\HP\OneDrive\Desktop\burnout_depression_project\models\burnout_depression_model.pkl")
    scaler = joblib.load(r"C:\Users\HP\OneDrive\Desktop\burnout_depression_project\models\scaler.pkl")
    le_gender = joblib.load(r"C:\Users\HP\OneDrive\Desktop\burnout_depression_project\models\gender_encoder.pkl")
    le_education = joblib.load(r"C:\Users\HP\OneDrive\Desktop\burnout_depression_project\models\education_encoder.pkl")
    le_employment = joblib.load(r"C:\Users\HP\OneDrive\Desktop\burnout_depression_project\models\employment_encoder.pkl")
    le_substances = joblib.load(r"C:\Users\HP\OneDrive\Desktop\burnout_depression_project\models\substance_encoder.pkl")
    feature_columns = joblib.load(r"C:\Users\HP\OneDrive\Desktop\burnout_depression_project\models\feature_columns.pkl")

    print("✓ All models loaded successfully!")
except Exception as e:
    print(f"✗ Error loading models: {e}")
    print("\nPlease run the training notebook first!")
    exit()

# Initialize RAG
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.Client()

def clean_text(text):
    """Clean and preprocess text"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

def analyze_user(user_data):
    """Complete analysis for a user"""
    
    # Preprocess
    cleaned_text = clean_text(user_data['Text'])
    text_length = len(cleaned_text.split())
    char_count = len(user_data['Text'])
    
    # Encode
    try:
        gender_enc = le_gender.transform([user_data['Gender']])[0]
        education_enc = le_education.transform([user_data['Education_Level']])[0]
        employment_enc = le_employment.transform([user_data['Employment_Status']])[0]
        substance_enc = le_substance.transform([user_data['Substance_Use']])[0]
    except:
        gender_enc = education_enc = employment_enc = substance_enc = 0
    
    # Create features
    features = pd.DataFrame({
        'Age': [user_data['Age']],
        'Gender_Encoded': [gender_enc],
        'Education_Encoded': [education_enc],
        'Employment_Encoded': [employment_enc],
        'Sleep_Hours': [user_data['Sleep_Hours']],
        'Anxiety_Score': [user_data['Anxiety_Score']],
        'Stress_Level': [user_data['Stress_Level']],
        'Family_History_Mental_Illness': [user_data['Family_History_Mental_Illness']],
        'Chronic_Illnesses': [user_data['Chronic_Illnesses']],
        'Substance_Encoded': [substance_enc],
        'Financial_Stress': [user_data['Financial_Stress']],
        'Work_Stress': [user_data['Work_Stress']],
        'Self_Esteem_Score': [user_data['Self_Esteem_Score']],
        'Loneliness_Score': [user_data['Loneliness_Score']],
        'Text_Length': [text_length],
        'Char_Count': [char_count]
    })
    
    # Predict
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    # Results
    print("\n" + "="*70)
    print("🏥 MENTAL HEALTH ANALYSIS REPORT")
    print("="*70)
    
    print(f"\n📋 USER INFORMATION:")
    print(f"   Name/ID: {user_data.get('name', 'Anonymous')}")
    print(f"   Age: {user_data['Age']}")
    print(f"   Gender: {user_data['Gender']}")
    print(f"   Employment: {user_data['Employment_Status']}")
    
    print(f"\n📊 KEY METRICS:")
    print(f"   Sleep Hours: {user_data['Sleep_Hours']}/night")
    print(f"   Anxiety Score: {user_data['Anxiety_Score']}/20")
    print(f"   Stress Level: {user_data['Stress_Level']}/10")
    print(f"   Self-Esteem: {user_data['Self_Esteem_Score']}/10")
    print(f"   Loneliness: {user_data['Loneliness_Score']}/10")
    
    print(f"\n🎯 DIAGNOSIS:")
    print(f"   Mental Health Status: {prediction}")
    
    print(f"\n📈 CONFIDENCE SCORES:")
    classes = model.classes_
    for i, cls in enumerate(classes):
        bar = "█" * int(probability[i] * 50)
        print(f"   {cls:12s}: {bar} {probability[i]*100:.1f}%")
    
    # Risk Level
    print(f"\n⚠️  RISK ASSESSMENT:")
    if prediction == 'Both':
        print("   🔴 HIGH RISK - Both Burnout and Depression Detected")
        print("   → Immediate professional help strongly recommended")
        print("   → Contact: Mental health crisis helpline")
    elif prediction == 'Depression':
        print("   🟡 MODERATE RISK - Depression Indicators Detected")
        print("   → Professional consultation recommended")
        print("   → Consider therapy or counseling")
    elif prediction == 'Burnout':
        print("   🟡 MODERATE RISK - Burnout Indicators Detected")
        print("   → Stress management needed")
        print("   → Work-life balance adjustment recommended")
    else:
        print("   🟢 LOW RISK - Healthy Mental State")
        print("   → Continue maintaining healthy habits")
        print("   → Regular self-monitoring recommended")
    
    # Recommendations
    recommendations = {
        'Depression': [
            "Seek professional therapy (CBT recommended)",
            "Maintain 7-9 hours sleep schedule",
            "Exercise 30 min daily (walking, yoga)",
            "Practice mindfulness meditation",
            "Connect with support groups"
        ],
        'Burnout': [
            "Set work-life boundaries",
            "Take regular breaks during work",
            "Practice stress-relief techniques",
            "Delegate tasks when possible",
            "Plan vacation or time off"
        ],
        'Both': [
            "Urgent: Consult mental health specialist",
            "Consider temporary work leave",
            "Therapy + stress management combination",
            "Build strong support system",
            "Daily self-care routine essential"
        ],
        'Healthy': [
            "Maintain current healthy lifestyle",
            "Continue regular exercise",
            "Keep social connections active",
            "Practice preventive mental care",
            "Monitor for early warning signs"
        ]
    }
    
    print(f"\n💡 PERSONALIZED RECOMMENDATIONS:")
    for i, rec in enumerate(recommendations[prediction], 1):
        print(f"   {i}. {rec}")
    
    print(f"\n📞 EMERGENCY CONTACTS:")
    print("   • National Suicide Prevention: 988")
    print("   • Crisis Text Line: Text HOME to 741741")
    print("   • SAMHSA Helpline: 1-800-662-4357")
    
    print("\n" + "="*70)
    
    return prediction, probability

# ============================================
# TEST CASES
# ============================================

print("\n🧪 TESTING WITH MULTIPLE USER PROFILES\n")

# Test Case 1: High Risk - Both
print("\n" + "="*70)
print("TEST CASE 1: HIGH STRESS PROFESSIONAL")
print("="*70)

user1 = {
    'name': 'Test User 1',
    'Age': 32,
    'Gender': 'Female',
    'Education_Level': "Master's",
    'Employment_Status': 'Employed',
    'Sleep_Hours': 4.5,
    'Anxiety_Score': 18,
    'Stress_Level': 9,
    'Family_History_Mental_Illness': 1,
    'Chronic_Illnesses': 0,
    'Substance_Use': 'Occasional',
    'Financial_Stress': 9,
    'Work_Stress': 10,
    'Self_Esteem_Score': 2,
    'Loneliness_Score': 9,
    'Text': "I'm completely exhausted and overwhelmed. Work is consuming my life and I can't sleep. I feel hopeless and anxious all the time. Nothing brings me joy anymore and I'm constantly stressed about deadlines and finances."
}

result1, prob1 = analyze_user(user1)

# Test Case 2: Burnout
print("\n" + "="*70)
print("TEST CASE 2: WORK BURNOUT")
print("="*70)

user2 = {
    'name': 'Test User 2',
    'Age': 28,
    'Gender': 'Male',
    'Education_Level': "Bachelor's",
    'Employment_Status': 'Employed',
    'Sleep_Hours': 6.0,
    'Anxiety_Score': 8,
    'Stress_Level': 9,
    'Family_History_Mental_Illness': 0,
    'Chronic_Illnesses': 0,
    'Substance_Use': 'none',
    'Financial_Stress': 6,
    'Work_Stress': 9,
    'Self_Esteem_Score': 6,
    'Loneliness_Score': 4,
    'Text': "Work has been really demanding lately. I'm constantly stressed about deadlines and feel exhausted. I need better work-life balance but don't know how to achieve it."
}

result2, prob2 = analyze_user(user2)

# Test Case 3: Depression
print("\n" + "="*70)
print("TEST CASE 3: DEPRESSION SYMPTOMS")
print("="*70)

user3 = {
    'name': 'Test User 3',
    'Age': 45,
    'Gender': 'Female',
    'Education_Level': "Bachelor's",
    'Employment_Status': 'Unemployed',
    'Sleep_Hours': 5.5,
    'Anxiety_Score': 15,
    'Stress_Level': 6,
    'Family_History_Mental_Illness': 1,
    'Chronic_Illnesses': 1,
    'Substance_Use': 'none',
    'Financial_Stress': 8,
    'Work_Stress': 3,
    'Self_Esteem_Score': 3,
    'Loneliness_Score': 9,
    'Text': "I feel so sad and empty all the time. I've lost interest in everything I used to enjoy. I feel worthless and isolated. I struggle to get out of bed most days."
}

result3, prob3 = analyze_user(user3)

# Test Case 4: Healthy
print("\n" + "="*70)
print("TEST CASE 4: HEALTHY INDIVIDUAL")
print("="*70)

user4 = {
    'name': 'Test User 4',
    'Age': 30,
    'Gender': 'Male',
    'Education_Level': "Master's",
    'Employment_Status': 'Employed',
    'Sleep_Hours': 7.5,
    'Anxiety_Score': 3,
    'Stress_Level': 4,
    'Family_History_Mental_Illness': 0,
    'Chronic_Illnesses': 0,
    'Substance_Use': 'none',
    'Financial_Stress': 3,
    'Work_Stress': 4,
    'Self_Esteem_Score': 8,
    'Loneliness_Score': 2,
    'Text': "Life is going well. I have a good balance between work and personal time. I exercise regularly and have supportive friends and family. Feeling positive overall."
}

result4, prob4 = analyze_user(user4)


# Interactive Testing
print("\n\n" + "="*70)
print("INTERACTIVE TESTING MODE")
print("="*70)

test_more = input("\nWould you like to test with custom input? (yes/no): ").lower()

if test_more == 'yes':
    print("\nPlease enter the following information:")
    
    custom_user = {
        'Age': int(input("Age: ")),
        'Gender': input("Gender (Male/Female/Non-Binary/Other): "),
        'Education_Level': input("Education Level (High School/Bachelor's/Master's/PhD/Other): "),
        'Employment_Status': input("Employment Status (Employed/Unemployed/Retired/Student): "),
        'Sleep_Hours': float(input("Average sleep hours per night: ")),
        'Anxiety_Score': int(input("Anxiety Score (0-20): ")),
        'Depression_Score': int(input("Depression Score (0-20): ")),
        'Stress_Level': int(input("Stress Level (0-10): ")),
        'Family_History_Mental_Illness': int(input("Family History of Mental Illness (0=No, 1=Yes): ")),
        'Chronic_Illnesses': int(input("Chronic Illnesses (0=No, 1=Yes): ")),
        'Substance_Use': input("Substance Use (none/Occasional/Frequent): "),
        'Financial_Stress': int(input("Financial Stress (0-10): ")),
        'Work_Stress': int(input("Work Stress (0-10): ")),
        'Self_Esteem_Score': int(input("Self-Esteem Score (0-10): ")),
        'Loneliness_Score': int(input("Loneliness Score (0-10): ")),
        'Text': input("Describe how you're feeling: ")
    }
    
    print("\n\n👤 CUSTOM USER TEST")
    result, prob = analyze_user(custom_user)

print("\n" + "="*70)
print("✅ ALL TESTS COMPLETED!")
print("="*70)
print("\nSummary:")
print(f"Test 1: {result1} - Expected: Both/Depression")
print(f"Test 2: {result2} - Expected: Burnout")
print(f"Test 3: {result3} - Expected: Depression")
print(f"Test 4: {result4} - Expected: Healthy")
print("\n" + "="*70)
