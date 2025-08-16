import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

class TreatmentResponseModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        
    def build_transformer_model(self, input_size, num_classes=2):
        """Build a transformer-based model for treatment response prediction"""
        class TreatmentResponseTransformer(nn.Module):
            def __init__(self, input_size, hidden_size=256, num_classes=2):
                super(TreatmentResponseTransformer, self).__init__()
                
                # Feature processing layers
                self.feature_encoder = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                )
                
                # Classification head
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_size // 2, hidden_size // 4),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_size // 4, num_classes)
                )
                
            def forward(self, x):
                features = self.feature_encoder(x)
                output = self.classifier(features)
                return output
        
        self.model = TreatmentResponseTransformer(input_size, num_classes=num_classes)
        return self.model
    
    def preprocess_data(self, df):
        """Preprocess the data for treatment response prediction"""
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
    
    def create_response_target(self, df, treatment_type=None):
        """Create treatment response target based on survival improvement"""
        # For now, create a synthetic response target
        # In practice, this would be based on actual treatment response data
        
        if treatment_type:
            # Filter by treatment type
            treatment_data = df[df['Treatment'] == treatment_type]
        else:
            treatment_data = df
        
        # Create synthetic response based on survival and other factors
        # Higher survival and better performance status = better response
        response_score = (
            treatment_data['Survival_Months'] / 100 +  # Normalize survival
            (5 - treatment_data['Performance_Status']) / 5 +  # Better performance = higher score
            treatment_data['Hemoglobin_Level'] / 20 +  # Normalize hemoglobin
            np.random.normal(0, 0.1, len(treatment_data))  # Add some randomness
        )
        
        # Convert to binary response (0: Poor, 1: Good)
        response_threshold = response_score.median()
        response_target = (response_score > response_threshold).astype(int)
        
        return response_target
    
    def train_model(self, df, treatment_type=None, epochs=50, batch_size=32):
        """Train the treatment response model"""
        try:
            # Preprocess data
            X = self.preprocess_data(df)
            
            # Create response target
            y = self.create_response_target(df, treatment_type)
            
            # Ensure X is numeric and convert to numpy array if needed
            if hasattr(X, 'values'):
                X = X.values
            X = X.astype(float)
            
            # Ensure y is numeric and convert to numpy array if needed
            if hasattr(y, 'values'):
                y = y.values
            y = y.astype(int)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Build model
            input_size = X_scaled.shape[1]
            self.build_transformer_model(input_size)
            
            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.LongTensor(y_train)
            X_test_tensor = torch.FloatTensor(X_test)
            y_test_tensor = torch.LongTensor(y_test)
            
            # Training setup
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            
            # Training loop
            self.model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = self.model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            
            # Evaluate
            self.model.eval()
            with torch.no_grad():
                test_outputs = self.model(X_test_tensor)
                _, predicted = torch.max(test_outputs.data, 1)
                accuracy = accuracy_score(y_test_tensor, predicted)
            
            print(f'Test Accuracy: {accuracy:.4f}')
            
            return {
                'model': self.model,
                'accuracy': accuracy,
                'feature_columns': self.feature_columns
            }
        except Exception as e:
            print(f"Error in train_model: {e}")
            raise
    
    def predict_response(self, patient_data, treatment_type):
        """Predict treatment response for a patient"""
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Preprocess patient data
        if isinstance(patient_data, dict):
            patient_df = pd.DataFrame([patient_data])
        else:
            patient_df = patient_data
        
        # Add treatment type if not present
        if 'Treatment' not in patient_df.columns:
            patient_df['Treatment'] = treatment_type
        
        # Preprocess
        X = self.preprocess_data(patient_df)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
        
        return {
            'response_class': int(predicted[0]),  # 0: Poor, 1: Good
            'good_response_probability': float(probabilities[0][1]),
            'poor_response_probability': float(probabilities[0][0]),
            'treatment_type': treatment_type
        }
    
    def save_model(self, model_dir):
        """Save the trained model"""
        os.makedirs(model_dir, exist_ok=True)
        
        if self.model is not None:
            torch.save(self.model.state_dict(), os.path.join(model_dir, 'treatment_response_model.pth'))
        
        joblib.dump(self.scaler, os.path.join(model_dir, 'treatment_scaler.pkl'))
        joblib.dump(self.label_encoders, os.path.join(model_dir, 'treatment_label_encoders.pkl'))
        joblib.dump(self.feature_columns, os.path.join(model_dir, 'treatment_feature_columns.pkl'))
    
    def load_model(self, model_dir, input_size):
        """Load the trained model"""
        model_path = os.path.join(model_dir, 'treatment_response_model.pth')
        scaler_path = os.path.join(model_dir, 'treatment_scaler.pkl')
        encoders_path = os.path.join(model_dir, 'treatment_label_encoders.pkl')
        features_path = os.path.join(model_dir, 'treatment_feature_columns.pkl')
        
        if os.path.exists(model_path):
            self.build_transformer_model(input_size)
            self.model.load_state_dict(torch.load(model_path))
        
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        
        if os.path.exists(encoders_path):
            self.label_encoders = joblib.load(encoders_path)
        
        if os.path.exists(features_path):
            self.feature_columns = joblib.load(features_path)
        
        return True

def create_treatment_patient_data(age, gender, stage, tumor_size, performance_status,
                                 hemoglobin, albumin, creatinine, treatment_type):
    """Create patient data dictionary for treatment response prediction"""
    return {
        'Age': age,
        'Gender': gender,
        'Stage': stage,
        'Tumor_Size_mm': tumor_size,
        'Treatment': treatment_type,
        'Performance_Status': performance_status,
        'Hemoglobin_Level': hemoglobin,
        'Albumin_Level': albumin,
        'Creatinine_Level': creatinine,
        # Add default values for missing features
        'Smoking_History': 'Former Smoker',
        'Tumor_Location': 'Upper Lobe',
        'Ethnicity': 'Caucasian',
        'Insurance_Type': 'Private',
        'Family_History': False,
        'Comorbidity_Diabetes': False,
        'Comorbidity_Hypertension': False,
        'Comorbidity_Heart_Disease': False,
        'Comorbidity_Chronic_Lung_Disease': False,
        'Comorbidity_Kidney_Disease': False,
        'Comorbidity_Autoimmune_Disease': False,
        'Comorbidity_Other': False,
        'Blood_Pressure_Systolic': 130,
        'Blood_Pressure_Diastolic': 85,
        'Blood_Pressure_Pulse': 80,
        'White_Blood_Cell_Count': 6.7,
        'Platelet_Count': 300,
        'Alkaline_Phosphatase_Level': 80,
        'Alanine_Aminotransferase_Level': 25,
        'Aspartate_Aminotransferase_Level': 30,
        'LDH_Level': 175,
        'Calcium_Level': 9.5,
        'Phosphorus_Level': 3.5,
        'Glucose_Level': 100,
        'Potassium_Level': 4.0,
        'Sodium_Level': 140,
        'Smoking_Pack_Years': 20
    } 