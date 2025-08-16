import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from lifelines import CoxPHFitter
import joblib
import os

class SurvivalPredictionModel:
    def __init__(self):
        self.svm_model = None
        self.cox_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        
    def preprocess_data(self, df):
        """Preprocess the data for survival prediction"""
        # Create a copy to avoid modifying original data
        data = df.copy()
        
        # Encode categorical variables
        categorical_columns = ['Gender', 'Smoking_History', 'Tumor_Location', 
                             'Stage', 'Treatment', 'Ethnicity', 'Insurance_Type',
                             'Family_History']
        
        for col in categorical_columns:
            if col in data.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    data[col] = self.label_encoders[col].fit_transform(data[col])
                else:
                    data[col] = self.label_encoders[col].transform(data[col])
        
        # Convert boolean columns to numeric
        boolean_columns = [col for col in data.columns if 'Comorbidity_' in col]
        for col in boolean_columns:
            if col in data.columns:
                # Handle both string ('Yes'/'No') and numeric (0/1) values
                if data[col].dtype == 'object':
                    data[col] = (data[col] == 'Yes').astype(int)
                else:
                    data[col] = data[col].astype(int)
        
        # Define feature columns
        feature_columns = [
            'Age', 'Gender', 'Smoking_History', 'Tumor_Size_mm', 'Tumor_Location',
            'Stage', 'Treatment', 'Ethnicity', 'Insurance_Type', 'Family_History',
            'Comorbidity_Diabetes', 'Comorbidity_Hypertension', 'Comorbidity_Heart_Disease',
            'Comorbidity_Chronic_Lung_Disease', 'Comorbidity_Kidney_Disease',
            'Comorbidity_Autoimmune_Disease', 'Comorbidity_Other', 'Performance_Status',
            'Blood_Pressure_Systolic', 'Blood_Pressure_Diastolic', 'Blood_Pressure_Pulse',
            'Hemoglobin_Level', 'White_Blood_Cell_Count', 'Platelet_Count',
            'Albumin_Level', 'Alkaline_Phosphatase_Level', 'Alanine_Aminotransferase_Level',
            'Aspartate_Aminotransferase_Level', 'Creatinine_Level', 'LDH_Level',
            'Calcium_Level', 'Phosphorus_Level', 'Glucose_Level', 'Potassium_Level',
            'Sodium_Level', 'Smoking_Pack_Years'
        ]
        
        # Filter columns that exist in the dataset
        self.feature_columns = [col for col in feature_columns if col in data.columns]
        
        # Get the data for selected features
        result_data = data[self.feature_columns].copy()
        
        # Ensure all columns are numeric
        for col in result_data.columns:
            if result_data[col].dtype == 'object':
                result_data[col] = pd.to_numeric(result_data[col], errors='coerce')
        
        # Fill any NaN values with 0
        result_data = result_data.fillna(0)
        
        return result_data
    
    def create_survival_target(self, df, threshold=None):
        """Create binary survival target based on median or threshold"""
        if threshold is None:
            threshold = df['Survival_Months'].median()
        
        return (df['Survival_Months'] > threshold).astype(int)
    
    def train_svm_model(self, X, y):
        """Train SVM model for survival classification"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train SVM
        self.svm_model = SVC(probability=True, random_state=42)
        self.svm_model.fit(X_scaled, y)
        
        return self.svm_model
    
    def train_cox_model(self, df):
        """Train Cox Proportional Hazards model"""
        try:
            # Use the preprocessed data instead of the original DataFrame
            # Get the preprocessed features
            X = self.preprocess_data(df)
            
            # Create cox_data with preprocessed features and survival info
            cox_data = X.copy()
            cox_data['Survival_Months'] = df['Survival_Months'].values
            cox_data['Event'] = 1
            
            # Train Cox model
            self.cox_model = CoxPHFitter()
            self.cox_model.fit(cox_data, duration_col='Survival_Months', event_col='Event')
            
            return self.cox_model
        except Exception as e:
            print(f"Error in train_cox_model: {e}")
            raise
    
    def predict_survival_svm(self, patient_data):
        """Predict survival using SVM model"""
        if self.svm_model is None:
            raise ValueError("SVM model not trained. Please train the model first.")
        
        # Preprocess patient data
        if isinstance(patient_data, dict):
            # Convert dict to DataFrame
            patient_df = pd.DataFrame([patient_data])
        else:
            patient_df = patient_data
        
        # Preprocess
        X = self.preprocess_data(patient_df)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.svm_model.predict(X_scaled)[0]
        probability = self.svm_model.predict_proba(X_scaled)[0]
        
        return {
            'survival_class': int(prediction),  # 0: Low survival, 1: High survival
            'high_survival_probability': float(probability[1]),
            'low_survival_probability': float(probability[0])
        }
    
    def predict_survival_cox(self, patient_data):
        """Predict survival time using Cox model"""
        if self.cox_model is None:
            raise ValueError("Cox model not trained. Please train the model first.")
        
        # Preprocess patient data
        if isinstance(patient_data, dict):
            patient_df = pd.DataFrame([patient_data])
        else:
            patient_df = patient_data
        
        # Prepare data for Cox prediction
        cox_data = self.preprocess_data(patient_df)
        cox_data['Survival_Months'] = 60  # Placeholder
        cox_data['Event'] = 1
        
        # Predict survival time
        predicted_survival = self.cox_model.predict_survival_function(cox_data)
        
        return {
            'predicted_survival_months': float(predicted_survival.iloc[0, 0]),
            'survival_curve': predicted_survival
        }
    
    def train_models(self, df):
        """Train both SVM and Cox models"""
        try:
            # Preprocess data
            X = self.preprocess_data(df)
            y = self.create_survival_target(df)
            
            # Ensure X is numeric
            X = X.astype(float)
            
            # Train SVM model
            self.train_svm_model(X, y)
            
            # Train Cox model
            self.train_cox_model(df)
            
            return {
                'svm_model': self.svm_model,
                'cox_model': self.cox_model,
                'feature_columns': self.feature_columns
            }
        except Exception as e:
            print(f"Error in train_models: {e}")
            raise
    
    def save_models(self, model_dir):
        """Save trained models"""
        os.makedirs(model_dir, exist_ok=True)
        
        if self.svm_model is not None:
            joblib.dump(self.svm_model, os.path.join(model_dir, 'svm_model.pkl'))
        
        if self.cox_model is not None:
            joblib.dump(self.cox_model, os.path.join(model_dir, 'cox_model.pkl'))
        
        joblib.dump(self.scaler, os.path.join(model_dir, 'scaler.pkl'))
        joblib.dump(self.label_encoders, os.path.join(model_dir, 'label_encoders.pkl'))
        joblib.dump(self.feature_columns, os.path.join(model_dir, 'feature_columns.pkl'))
    
    def load_models(self, model_dir):
        """Load trained models"""
        svm_path = os.path.join(model_dir, 'svm_model.pkl')
        cox_path = os.path.join(model_dir, 'cox_model.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        encoders_path = os.path.join(model_dir, 'label_encoders.pkl')
        features_path = os.path.join(model_dir, 'feature_columns.pkl')
        
        if os.path.exists(svm_path):
            self.svm_model = joblib.load(svm_path)
        
        if os.path.exists(cox_path):
            self.cox_model = joblib.load(cox_path)
        
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        
        if os.path.exists(encoders_path):
            self.label_encoders = joblib.load(encoders_path)
        
        if os.path.exists(features_path):
            self.feature_columns = joblib.load(features_path)
        
        return True

def create_patient_data(age, gender, smoking_history, tumor_size, tumor_location,
                       stage, treatment, performance_status, hemoglobin, wbc_count,
                       diabetes=False, hypertension=False, heart_disease=False,
                       lung_disease=False, kidney_disease=False, autoimmune=False,
                       other_comorbidity=False, family_history=False):
    """Create patient data dictionary from input parameters"""
    return {
        'Age': age,
        'Gender': gender,
        'Smoking_History': smoking_history,
        'Tumor_Size_mm': tumor_size,
        'Tumor_Location': tumor_location,
        'Stage': stage,
        'Treatment': treatment,
        'Performance_Status': performance_status,
        'Hemoglobin_Level': hemoglobin,
        'White_Blood_Cell_Count': wbc_count,
        'Comorbidity_Diabetes': diabetes,
        'Comorbidity_Hypertension': hypertension,
        'Comorbidity_Heart_Disease': heart_disease,
        'Comorbidity_Chronic_Lung_Disease': lung_disease,
        'Comorbidity_Kidney_Disease': kidney_disease,
        'Comorbidity_Autoimmune_Disease': autoimmune,
        'Comorbidity_Other': other_comorbidity,
        'Family_History': family_history,
        # Add default values for missing features
        'Ethnicity': 'Caucasian',
        'Insurance_Type': 'Private',
        'Blood_Pressure_Systolic': 130,
        'Blood_Pressure_Diastolic': 85,
        'Blood_Pressure_Pulse': 80,
        'Platelet_Count': 300,
        'Albumin_Level': 4.0,
        'Alkaline_Phosphatase_Level': 80,
        'Alanine_Aminotransferase_Level': 25,
        'Aspartate_Aminotransferase_Level': 30,
        'Creatinine_Level': 1.0,
        'LDH_Level': 175,
        'Calcium_Level': 9.5,
        'Phosphorus_Level': 3.5,
        'Glucose_Level': 100,
        'Potassium_Level': 4.0,
        'Sodium_Level': 140,
        'Smoking_Pack_Years': 20
    } 