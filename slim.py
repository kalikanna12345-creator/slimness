import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os

# Page configuration
st.set_page_config(
    page_title="Slimness Prediction & Health Analysis",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #ffffff;
        text-align: center;
        font-size: 3em;
        margin-bottom: 0.5rem;
    }
    h2, h3 {
        color: #667eea;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.2);
        border-radius: 10px;
        color: white;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: white;
        color: #667eea;
    }
    div[data-testid="stExpander"] {
        background-color: white;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'patient_records' not in st.session_state:
    st.session_state.patient_records = []

# Data storage file
DATA_FILE = 'patient_records.json'

# Load patient records
def load_patient_records():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

# Save patient record
def save_patient_record(record):
    records = load_patient_records()
    records.append(record)
    with open(DATA_FILE, 'w') as f:
        json.dump(records, f, indent=2)

# Train model function
@st.cache_resource
def train_model(df):
    """Train the Random Forest model"""
    dataset_df = df.copy()
    
    categorical_columns = ['Gender', 'PhysicalActivity', 'FrequentConsumptionHighCalorieFood', 
                          'FrequentVegetableConsumption']
    
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        dataset_df[col] = le.fit_transform(dataset_df[col])
        label_encoders[col] = le
    
    target_encoder = LabelEncoder()
    dataset_df['Category_encoded'] = target_encoder.fit_transform(dataset_df['Category'])
    
    feature_columns = [
        'Height_m', 'Weight_kg', 'Age', 'Gender', 'PhysicalActivity',
        'FrequentConsumptionHighCalorieFood', 'FrequentVegetableConsumption',
        'BMI', 'Water_Intake_L', 'Sleep_Hours', 'Sleep_Quality_Score',
        'Screen_Time_Hours', 'Steps_Per_Day', 'Protein_Intake_g', 'Stress_Level_Score'
    ]
    
    X = dataset_df[feature_columns]
    y = dataset_df['Category_encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    rf_multi = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_multi.fit(X_train_scaled, y_train)
    
    y_pred = rf_multi.predict(scaler.transform(X_test))
    accuracy = accuracy_score(y_test, y_pred)
    
    return rf_multi, scaler, label_encoders, target_encoder, feature_columns, accuracy, dataset_df

# Generate visualizations
def generate_pie_charts(df):
    """Generate pie charts for data distribution"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor('white')
    
    # 1. Category Distribution
    category_counts = df['Category'].value_counts()
    colors1 = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#ffa726']
    axes[0, 0].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
                   startangle=90, colors=colors1, textprops={'fontsize': 11, 'weight': 'bold'})
    axes[0, 0].set_title('Category Distribution', fontsize=14, fontweight='bold')
    
    # 2. Gender Distribution
    gender_counts = df['Gender'].value_counts()
    colors2 = ['#667eea', '#764ba2']
    axes[0, 1].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%',
                   startangle=90, colors=colors2, textprops={'fontsize': 11, 'weight': 'bold'})
    axes[0, 1].set_title('Gender Distribution', fontsize=14, fontweight='bold')
    
    # 3. Physical Activity Distribution
    activity_counts = df['PhysicalActivity'].value_counts()
    colors3 = ['#ff6b6b', '#ffa726', '#66bb6a', '#42a5f5']
    axes[0, 2].pie(activity_counts.values, labels=activity_counts.index, autopct='%1.1f%%',
                   startangle=90, colors=colors3, textprops={'fontsize': 11, 'weight': 'bold'})
    axes[0, 2].set_title('Physical Activity Levels', fontsize=14, fontweight='bold')
    
    # 4. High Calorie Food Consumption
    calorie_counts = df['FrequentConsumptionHighCalorieFood'].value_counts()
    colors4 = ['#66bb6a', '#ff6b6b']
    axes[1, 0].pie(calorie_counts.values, labels=['Rarely', 'Frequently'], autopct='%1.1f%%',
                   startangle=90, colors=colors4, textprops={'fontsize': 11, 'weight': 'bold'})
    axes[1, 0].set_title('High Calorie Food Consumption', fontsize=14, fontweight='bold')
    
    # 5. Vegetable Consumption
    veg_counts = df['FrequentVegetableConsumption'].value_counts()
    colors5 = ['#66bb6a', '#ffa726']
    axes[1, 1].pie(veg_counts.values, labels=['Regularly', 'Rarely'], autopct='%1.1f%%',
                   startangle=90, colors=colors5, textprops={'fontsize': 11, 'weight': 'bold'})
    axes[1, 1].set_title('Vegetable Consumption', fontsize=14, fontweight='bold')
    
    # 6. Age Groups
    age_bins = [0, 20, 30, 40, 50, 100]
    age_labels = ['0-20', '21-30', '31-40', '41-50', '50+']
    df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
    age_group_counts = df['AgeGroup'].value_counts()
    colors6 = ['#ff6b6b', '#ffa726', '#66bb6a', '#42a5f5', '#ab47bc']
    axes[1, 2].pie(age_group_counts.values, labels=age_group_counts.index, autopct='%1.1f%%',
                   startangle=90, colors=colors6, textprops={'fontsize': 11, 'weight': 'bold'})
    axes[1, 2].set_title('Age Group Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

# Get suggestions
def get_suggestions(category, user_data):
    suggestions = []
    action_plan = []
    
    if category == 'Underweight':
        suggestions.append("Increase calorie intake by 300-500 calories daily")
        suggestions.append("Focus on protein-rich meals (aim for 1.2-1.5g per kg body weight)")
        suggestions.append("Include healthy fats like nuts, avocados, and olive oil")
        action_plan.extend([
            "Week 1-2: Increase protein intake to 70-80g daily",
            "Week 3-4: Add strength training 2-3 times per week",
            "Month 2: Consult nutritionist for personalized meal plan"
        ])
    elif category == 'Overweight' or category == 'Obese':
        suggestions.append("Increase daily steps gradually to 10,000")
        suggestions.append("Reduce high-calorie food consumption")
        suggestions.append("Practice portion control and mindful eating")
        action_plan.extend([
            "Week 1-2: Walk 30 minutes daily, track food intake",
            "Week 3-4: Add meal planning and food prep",
            "Month 2: Incorporate resistance training 2x per week"
        ])
    elif category == 'Healthy Slim':
        suggestions.append("Maintain current healthy habits")
        suggestions.append("Continue regular physical activity")
        action_plan.append("Monitor weight monthly to maintain healthy range")
    
    if user_data['sleep'] < 7:
        suggestions.append(f"Increase sleep from {user_data['sleep']} to 7-9 hours per night")
        action_plan.append("Establish consistent bedtime routine")
    
    if user_data['water'] < 2:
        suggestions.append("Increase water intake to 2-3L daily")
        action_plan.append("Set hourly water reminders on your phone")
    
    if user_data['stress'] > 6:
        suggestions.append("Implement stress management techniques")
        action_plan.append("Practice meditation or deep breathing exercises 10 min daily")
    
    if user_data['screenTime'] > 6:
        suggestions.append("Reduce screen time to improve sleep quality")
    
    return suggestions, action_plan

# Main App
def main():
    # Header
    st.markdown("<h1>🏃‍♂️ Slimness Prediction & Health Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: white; font-size: 1.2em; margin-bottom: 2rem;'>AI-Powered Health Assessment & Analytics</p>", unsafe_allow_html=True)
    
    # Load dataset
    try:
        df = pd.read_csv('augmented_obesity_lifestyle_dataset (1).csv')
        
        # Train model
        if not st.session_state.model_trained:
            with st.spinner('🤖 Training AI model...'):
                model, scaler, label_encoders, target_encoder, feature_columns, accuracy, dataset_df = train_model(df)
                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.label_encoders = label_encoders
                st.session_state.target_encoder = target_encoder
                st.session_state.feature_columns = feature_columns
                st.session_state.accuracy = accuracy
                st.session_state.dataset_df = dataset_df
                st.session_state.df_original = df
                st.session_state.model_trained = True
    except FileNotFoundError:
        st.error("❌ Dataset file 'augmented_obesity_lifestyle_dataset (1).csv' not found!")
        st.info("Please upload the dataset file to continue.")
        uploaded_file = st.file_uploader("Upload CSV Dataset", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            df.to_csv('augmented_obesity_lifestyle_dataset (1).csv', index=False)
            st.success("✅ Dataset uploaded successfully! Please refresh the page.")
            st.stop()
        st.stop()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📊 Data Visualization", "📋 Dataset"])
    
    # Tab 1: Prediction
    with tab1:
        st.markdown("<div style='background: white; padding: 30px; border-radius: 15px;'>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 👤 Patient Information")
            name = st.text_input("📛 Full Name", key="name")
            email = st.text_input("📧 Email", key="email")
            phone = st.text_input("📞 Phone", key="phone")
            date = st.date_input("📅 Date", datetime.now(), key="date")
            
            st.markdown("### 📊 Basic Metrics")
            height = st.number_input("📏 Height (m)", min_value=1.0, max_value=2.5, value=1.75, step=0.01)
            weight = st.number_input("⚖️ Weight (kg)", min_value=30.0, max_value=200.0, value=68.0, step=0.1)
            age = st.number_input("🎂 Age", min_value=10, max_value=100, value=28, step=1)
            gender = st.selectbox("👤 Gender", ["Male", "Female"])
            
            st.markdown("### 🏃 Activity & Lifestyle")
            activity = st.selectbox("Physical Activity", ["Sedentary", "Light", "Moderate", "High"])
            steps = st.number_input("🚶 Daily Steps", min_value=0, max_value=50000, value=8500, step=100)
            screenTime = st.number_input("📱 Screen Time (hours)", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
        
        with col2:
            st.markdown("### 🍽️ Nutrition")
            protein = st.number_input("🥩 Protein (g/day)", min_value=0, max_value=300, value=75, step=5)
            water = st.number_input("💧 Water (L/day)", min_value=0.0, max_value=10.0, value=2.5, step=0.1)
            highCalorie = st.selectbox("🍔 High-Calorie Food", ["no", "yes"])
            vegetables = st.selectbox("🥦 Vegetables", ["yes", "no"])
            
            st.markdown("### 😴 Sleep & Wellness")
            sleep = st.number_input("😴 Sleep (hours)", min_value=0.0, max_value=24.0, value=7.5, step=0.5)
            sleepQuality = st.slider("⭐ Sleep Quality (1-10)", 1, 10, 7)
            stress = st.slider("🧠 Stress Level (1-10)", 1, 10, 4)
        
        if st.button("🔍 Analyze Health Status", use_container_width=True):
            if not name or not email or not phone:
                st.warning("⚠️ Please fill in all patient information fields!")
            else:
                with st.spinner('Analyzing your health profile...'):
                    # Calculate BMI
                    bmi = weight / (height ** 2)
                    
                    # Prepare user input
                    user_input = []
                    for col in st.session_state.feature_columns:
                        if col == 'BMI':
                            user_input.append(bmi)
                        elif col == 'Height_m':
                            user_input.append(height)
                        elif col == 'Weight_kg':
                            user_input.append(weight)
                        elif col == 'Age':
                            user_input.append(age)
                        elif col == 'Gender':
                            user_input.append(st.session_state.label_encoders['Gender'].transform([gender])[0])
                        elif col == 'PhysicalActivity':
                            user_input.append(st.session_state.label_encoders['PhysicalActivity'].transform([activity])[0])
                        elif col == 'FrequentConsumptionHighCalorieFood':
                            user_input.append(st.session_state.label_encoders['FrequentConsumptionHighCalorieFood'].transform([highCalorie])[0])
                        elif col == 'FrequentVegetableConsumption':
                            user_input.append(st.session_state.label_encoders['FrequentVegetableConsumption'].transform([vegetables])[0])
                        elif col == 'Water_Intake_L':
                            user_input.append(water)
                        elif col == 'Sleep_Hours':
                            user_input.append(sleep)
                        elif col == 'Sleep_Quality_Score':
                            user_input.append(sleepQuality)
                        elif col == 'Screen_Time_Hours':
                            user_input.append(screenTime)
                        elif col == 'Steps_Per_Day':
                            user_input.append(steps)
                        elif col == 'Protein_Intake_g':
                            user_input.append(protein)
                        elif col == 'Stress_Level_Score':
                            user_input.append(stress)
                    
                    # Make prediction
                    user_input_scaled = st.session_state.scaler.transform([user_input])
                    prediction = st.session_state.model.predict(user_input_scaled)[0]
                    probabilities = st.session_state.model.predict_proba(user_input_scaled)[0]
                    
                    category = st.session_state.target_encoder.inverse_transform([prediction])[0]
                    confidence = probabilities[prediction]
                    
                    # BMI Category
                    if bmi < 18.5:
                        bmi_category = "Underweight"
                    elif 18.5 <= bmi < 25:
                        bmi_category = "Normal weight"
                    elif 25 <= bmi < 30:
                        bmi_category = "Overweight"
                    else:
                        bmi_category = "Obese"
                    
                    # Feature importance
                    importances = st.session_state.model.feature_importances_
                    top_indices = np.argsort(importances)[-3:][::-1]
                    top_features = []
                    for idx in top_indices:
                        top_features.append({
                            'feature': st.session_state.feature_columns[idx],
                            'value': user_input[idx],
                            'importance': float(importances[idx])
                        })
                    
                    # Get suggestions
                    user_data_dict = {
                        'bmi': bmi,
                        'sleep': sleep,
                        'water': water,
                        'stress': stress,
                        'screenTime': screenTime,
                        'steps': steps,
                        'protein': protein
                    }
                    suggestions, action_plan = get_suggestions(category, user_data_dict)
                    
                    # Display results
                    st.success("✅ Analysis Complete!")
                    
                    st.markdown("---")
                    st.markdown(f"### 📊 Health Analysis Report for {name}")
                    st.markdown(f"**Date:** {date}")
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🎯 Prediction", category)
                    with col2:
                        st.metric("💪 Confidence", f"{confidence*100:.1f}%")
                    with col3:
                        st.metric("📊 BMI", f"{bmi:.1f} ({bmi_category})")
                    
                    # Top Features
                    with st.expander("🔍 Top Influencing Factors", expanded=True):
                        for feat in top_features:
                            st.write(f"**{feat['feature']}:** {feat['value']:.2f if isinstance(feat['value'], float) else feat['value']} (Importance: {feat['importance']*100:.1f}%)")
                    
                    # Recommendations
                    with st.expander("💡 Health Recommendations", expanded=True):
                        for sug in suggestions:
                            st.write(f"• {sug}")
                    
                    # Action Plan
                    if action_plan:
                        with st.expander("📅 Action Plan", expanded=True):
                            for plan in action_plan:
                                st.write(f"• {plan}")
                    
                    # Save record
                    record = {
                        'name': name,
                        'email': email,
                        'phone': phone,
                        'date': str(date),
                        'category': category,
                        'bmi': bmi,
                        'confidence': confidence,
                        'timestamp': datetime.now().isoformat()
                    }
                    save_patient_record(record)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Tab 2: Visualization
    with tab2:
        st.markdown("<div style='background: white; padding: 30px; border-radius: 15px;'>", unsafe_allow_html=True)
        
        st.markdown("### 📊 Dataset Distribution Analysis")
        
        # Statistics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Records", len(st.session_state.df_original))
        with col2:
            st.metric("Average Age", f"{st.session_state.df_original['Age'].mean():.1f}")
        with col3:
            st.metric("Average BMI", f"{st.session_state.df_original['BMI'].mean():.1f}")
        with col4:
            gender_counts = st.session_state.df_original['Gender'].value_counts()
            male_percent = (gender_counts.get('Male', 0) / len(st.session_state.df_original) * 100)
            st.metric("Male %", f"{male_percent:.1f}%")
        with col5:
            female_percent = (gender_counts.get('Female', 0) / len(st.session_state.df_original) * 100)
            st.metric("Female %", f"{female_percent:.1f}%")
        
        st.markdown("---")
        
        # Generate and display charts
        fig = generate_pie_charts(st.session_state.df_original)
        st.pyplot(fig)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Tab 3: Dataset
    with tab3:
        st.markdown("<div style='background: white; padding: 30px; border-radius: 15px;'>", unsafe_allow_html=True)
        
        st.markdown("### 📋 Training Dataset")
        
        # Display dataset
        st.dataframe(st.session_state.df_original.head(100), use_container_width=True)
        
        st.info(f"Showing first 100 of {len(st.session_state.df_original)} total records")
        
        # Download button
        csv = st.session_state.df_original.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Dataset",
            data=csv,
            file_name="obesity_dataset.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
