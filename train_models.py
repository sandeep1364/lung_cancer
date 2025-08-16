#!/usr/bin/env python3
"""
Training script for Lung Cancer Predictive Models
This script trains all three models and saves them for use in the Streamlit application.
"""

import os
import pandas as pd
import numpy as np
from models import (
    LungNoduleDetectionModel, 
    SurvivalPredictionModel, 
    TreatmentResponseModel,
    prepare_dataset,
    create_patient_data,
    create_treatment_patient_data
)

def train_lung_detection_model():
    """Train the lung nodule detection model"""
    print("🫁 Training Lung Nodule Detection Model...")
    
    # Initialize model
    detection_model = LungNoduleDetectionModel()
    
    # Check if dataset exists
    dataset_path = "data/archive (8)/The IQ-OTHNCCD lung cancer dataset/The IQ-OTHNCCD lung cancer dataset"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at {dataset_path}")
        print("Please ensure the image dataset is available.")
        return False
    
    try:
        # Prepare dataset
        print("📊 Preparing dataset...")
        X_train, X_test, y_train, y_test = prepare_dataset(dataset_path)
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Testing data shape: {X_test.shape}")
        
        # Build and train model
        print("🏗️ Building CNN model...")
        model = detection_model.build_model()
        
        print("🎯 Training model...")
        history = model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Save model
        model_dir = "trained_models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "lung_detection_model.h5")
        detection_model.save_model(model_path)
        
        print(f"✅ Lung detection model saved to {model_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error training lung detection model: {e}")
        return False

def train_survival_prediction_model():
    """Train the survival prediction model"""
    print("⏰ Training Survival Prediction Model...")
    
    # Initialize model
    survival_model = SurvivalPredictionModel()
    
    # Check if dataset exists
    csv_path = "data/archive (7)/lung_cancer_data.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ Dataset not found at {csv_path}")
        print("Please ensure the clinical dataset is available.")
        return False
    
    try:
        # Load data
        print("📊 Loading clinical data...")
        df = pd.read_csv(csv_path)
        print(f"Dataset shape: {df.shape}")
        
        # Train models
        print("🎯 Training SVM and Cox models...")
        try:
            results = survival_model.train_models(df)
            print("✅ Models trained successfully")
        except Exception as e:
            print(f"❌ Error during training: {e}")
            print("Attempting to continue with other models...")
            return False
        
        # Save models
        model_dir = "trained_models"
        os.makedirs(model_dir, exist_ok=True)
        survival_model.save_models(model_dir)
        
        print(f"✅ Survival prediction models saved to {model_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Error training survival prediction model: {e}")
        return False

def train_treatment_response_model():
    """Train the treatment response model"""
    print("💊 Training Treatment Response Model...")
    
    # Initialize model
    treatment_model = TreatmentResponseModel()
    
    # Check if dataset exists
    csv_path = "data/archive (7)/lung_cancer_data.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ Dataset not found at {csv_path}")
        print("Please ensure the clinical dataset is available.")
        return False
    
    try:
        # Load data
        print("📊 Loading clinical data...")
        df = pd.read_csv(csv_path)
        print(f"Dataset shape: {df.shape}")
        
        # Train model for each treatment type
        treatment_types = ["Surgery", "Radiation Therapy", "Chemotherapy", "Targeted Therapy"]
        
        for treatment_type in treatment_types:
            print(f"🎯 Training model for {treatment_type}...")
            
            # Filter data for this treatment type
            treatment_data = df[df['Treatment'] == treatment_type].copy()
            
            if len(treatment_data) > 100:  # Only train if we have enough data
                try:
                    # Create a new model instance for each treatment type
                    treatment_model_instance = TreatmentResponseModel()
                    results = treatment_model_instance.train_model(
                        treatment_data, 
                        treatment_type=treatment_type,
                        epochs=30  # Reduced epochs for faster training
                    )
                    print(f"✅ Model training completed for {treatment_type}")
                    
                    # Save model
                    model_dir = f"trained_models/treatment_{treatment_type.replace(' ', '_')}"
                    os.makedirs(model_dir, exist_ok=True)
                    treatment_model_instance.save_model(model_dir)
                    
                    print(f"✅ Treatment response model for {treatment_type} saved")
                except Exception as e:
                    print(f"❌ Error training model for {treatment_type}: {e}")
                    continue
            else:
                print(f"⚠️ Insufficient data for {treatment_type} ({len(treatment_data)} samples)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error training treatment response model: {e}")
        return False

def main():
    """Main training function"""
    print("🚀 Starting Lung Cancer Model Training")
    print("=" * 50)
    
    # Create models directory
    os.makedirs("trained_models", exist_ok=True)
    
    # Train models
    results = {}
    
    # Train lung detection model
    results['lung_detection'] = train_lung_detection_model()
    
    # Train survival prediction model
    results['survival_prediction'] = train_survival_prediction_model()
    
    # Train treatment response model
    results['treatment_response'] = train_treatment_response_model()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Training Summary:")
    print("=" * 50)
    
    for model_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{model_name}: {status}")
    
    successful_models = sum(results.values())
    total_models = len(results)
    
    print(f"\nOverall: {successful_models}/{total_models} models trained successfully")
    
    if successful_models == total_models:
        print("🎉 All models trained successfully! You can now run the Streamlit app.")
    else:
        print("⚠️ Some models failed to train. Check the error messages above.")
    
    print("\nTo run the Streamlit app:")
    print("streamlit run streamlit_app.py")

if __name__ == "__main__":
    main() 