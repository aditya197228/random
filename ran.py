import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    irrelevant_columns = ["id", "dataset"]
    df = df.drop(columns=[col for col in irrelevant_columns if col in df.columns], errors="ignore")
    df.fillna(df.median(numeric_only=True), inplace=True)
    
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    target_col = "target"#deburging
    if target_col not in df.columns:
        raise ValueError(f"Error: Column '{target_col}' not found in the dataset.")
    
    X = df.drop(columns=[target_col])#spplited the data
    y = df[target_col]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, X.columns, scaler

def train_model(X_scaled, y):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    
    joblib.dump(model, "heart_disease_model.pkl")
    return model

def get_user_input():
    user_data = {}
    input_details = {
        "age": "Enter your age (e.g., 45): ",
        "sex": "Enter sex (1 for male, 0 for female): ",
        "cp": "Enter chest pain type (0 = None, 1 = Mild, 2 = Moderate, 3 = Severe): ",
        "trestbps": "Enter resting blood pressure (mm Hg, e.g., 120): ",
        "chol": "Enter cholesterol level (mg/dL, e.g., 200): ",
        "fbs": "Enter fasting blood sugar (1 if > 120 mg/dL, otherwise 0): ",
        "restecg": "Enter ECG results (0 = Normal, 1 = ST-T abnormality, 2 = LV hypertrophy): ",
        "thalach": "Enter max heart rate achieved (e.g., 150): ",
        "exang": "Exercise-induced angina (1 = Yes, 0 = No): ",
        "oldpeak": "ST depression induced by exercise (e.g., 1.2): ",
        "slope": "Slope of peak exercise ST segment (0 = Upsloping, 1 = Flat, 2 = Downsloping): ",
        "ca": "Number of major vessels (0-4): ",
        "thal": "Enter thalassemia type (0 = Normal, 1 = Fixed defect, 2 = Reversible defect): "
    }
    
    for feature, prompt in input_details.items():#user input for the data
        user_data[feature] = float(input(prompt))
    
    return user_data

def assess_risk_factors(user_data):
    recommendations = []
    if user_data["chol"] > 240:
        recommendations.append("Reduce cholesterol by eating more fiber, reducing saturated fats, and exercising regularly.")
    if user_data["trestbps"] > 130:
        recommendations.append("Lower blood pressure through a low-sodium diet, regular exercise, and stress management.")
    if user_data["thalach"] < 100:
        recommendations.append("Increase cardiovascular activity to improve heart rate performance.")
    if not recommendations:
        recommendations.append("Maintain a balanced diet, stay active, and monitor your health regularly.")
    
    return recommendations

def predict_heart_disease(user_data, model, scaler, feature_names):
    input_array = np.array([user_data[feature] for feature in feature_names]).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]
    risk = "High" if prediction == 1 else "Low"
    recommendations = assess_risk_factors(user_data)
    
    return "Heart Disease Detected" if prediction == 1 else "No Heart Disease", risk, recommendations

def chatbot():
    print("\nHealth Chatbot: Ask me anything about heart disease prevention! Type 'exit' to quit.")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        elif "cholesterol" in query:
            print("Chatbot: To reduce cholesterol, eat more vegetables, whole grains, and healthy fats like olive oil.")
        elif "blood pressure" in query:
            print("Chatbot: Lower your blood pressure by reducing salt intake, exercising, and managing stress.")
        elif "exercise" in query:
            print("Chatbot: Engage in at least 30 minutes of moderate exercise daily for heart health.")
        else:
            print("Chatbot: I can provide tips on cholesterol, blood pressure, and general heart health.")

file_path = r"C:\Users\Aditya Chaudhary\Desktop\randome\heart.csv"
X_scaled, y, feature_names, scaler = load_and_preprocess_data(file_path)
model = train_model(X_scaled, y)
user_data = get_user_input()
prediction, risk_level, recommendations = predict_heart_disease(user_data, model, scaler, feature_names)

print("\n--- Heart Disease Prediction Results ---")
print(f"Prediction: {prediction}")
print(f"Risk Level: {risk_level}")
print("Recommendations:")
for rec in recommendations:
    print(f"- {rec}")

chatbot()