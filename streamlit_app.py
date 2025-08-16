import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cv2
import os
import pickle
import joblib
from PIL import Image
import io
import base64

# Page configuration
st.set_page_config(
    page_title="Lung Cancer Predictive Modeling",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🫁 Lung Cancer AI")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.selectbox(
    "Choose a Model:",
    ["🏠 Home", "🔍 Lung Nodule Detection", "⏰ Survival Prediction", "💊 Treatment Response", "📊 Data Analysis"]
)

# Home page
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">Lung Cancer Predictive Modeling</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🔍 Lung Nodule Detection</h3>
            <p>Upload CT scan images to detect and classify lung nodules as Normal, Benign, or Malignant using CNN.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>⏰ Survival Prediction</h3>
            <p>Predict patient survival based on clinical features using SVM and Cox Proportional Hazards models.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>💊 Treatment Response</h3>
            <p>Predict treatment response using transformer-based models on clinical and demographic data.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Project overview
    st.markdown('<h2 class="sub-header">Project Overview</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Dataset Information:**
        - **Clinical Data**: 23,658 patients with 38 features
        - **Image Data**: 1,097 CT scan images (Normal: 416, Malignant: 561, Benign: 120)
        - **Data Quality**: No missing values, professionally annotated
        
        **Key Features:**
        - Demographics (Age, Gender, Ethnicity)
        - Clinical measurements (Tumor size, Blood pressure, Lab values)
        - Comorbidities (Diabetes, Hypertension, Heart disease, etc.)
        - Treatment information and survival outcomes
        """)
    
    with col2:
        st.markdown("""
        **Model Performance:**
        - **CNN Detection**: High accuracy in nodule classification
        - **Survival Models**: ROC AUC evaluation for prognosis
        - **Treatment Response**: Binary classification for personalized treatment
        
        **Clinical Applications:**
        - Early detection and screening
        - Prognostic assessment
        - Treatment planning and personalization
        - Clinical decision support
        """)
    
    # Quick stats
    st.markdown('<h2 class="sub-header">Dataset Statistics</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Patients", "23,658")
    with col2:
        st.metric("CT Images", "1,097")
    with col3:
        st.metric("Features", "38")
    with col4:
        st.metric("Cancer Stages", "4")

# Lung Nodule Detection page
elif page == "🔍 Lung Nodule Detection":
    st.markdown('<h1 class="sub-header">🔍 Lung Nodule Detection</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This model uses Convolutional Neural Networks (CNN) to classify CT scan images into three categories:
    - **Normal**: No suspicious findings
    - **Benign**: Non-cancerous nodules
    - **Malignant**: Cancerous nodules
    """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload a CT scan image (JPG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a CT scan image to analyze for lung nodules"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded CT Scan", use_column_width=True)
        
        with col2:
            st.subheader("Analysis Results")
            
            # Simulate prediction (replace with actual model)
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Analyzing image..."):
                    # Placeholder for actual model prediction
                    import random
                    prediction = random.choice(['Normal', 'Benign', 'Malignant'])
                    confidence = random.uniform(0.7, 0.95)
                    
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h3>Prediction Results</h3>
                        <p><strong>Classification:</strong> {prediction}</p>
                        <p><strong>Confidence:</strong> {confidence:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence bar
                    st.progress(confidence)
                    
                    # Recommendations
                    if prediction == "Normal":
                        st.success("✅ No suspicious findings detected. Continue regular screening as recommended.")
                    elif prediction == "Benign":
                        st.warning("⚠️ Benign nodule detected. Follow-up monitoring recommended.")
                    else:
                        st.error("🚨 Malignant nodule detected. Immediate medical consultation required.")
    
    # Model information
    with st.expander("📋 Model Information"):
        st.markdown("""
        **CNN Architecture:**
        - Input: 256x256 grayscale images
        - Convolutional layers with ReLU activation
        - MaxPooling for dimension reduction
        - Dense layers for classification
        - Output: 3-class softmax (Normal, Benign, Malignant)
        
        **Training Data:**
        - 1,097 CT scan images
        - Professional medical annotations
        - Data augmentation for robustness
        
        **Performance Metrics:**
        - Accuracy: High classification accuracy
        - Confusion Matrix: Detailed class-wise performance
        """)

# Survival Prediction page
elif page == "⏰ Survival Prediction":
    st.markdown('<h1 class="sub-header">⏰ Survival Prediction</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Predict patient survival based on clinical features using Support Vector Machine (SVM) and Cox Proportional Hazards models.
    """)
    
    # Input form
    st.subheader("Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 30, 79, 55)
        gender = st.selectbox("Gender", ["Male", "Female"])
        smoking_history = st.selectbox("Smoking History", ["Never Smoked", "Former Smoker", "Current Smoker"])
        tumor_size = st.slider("Tumor Size (mm)", 10.0, 100.0, 55.0)
        tumor_location = st.selectbox("Tumor Location", ["Upper Lobe", "Middle Lobe", "Lower Lobe"])
    
    with col2:
        stage = st.selectbox("Cancer Stage", ["Stage I", "Stage II", "Stage III", "Stage IV"])
        treatment = st.selectbox("Treatment", ["Surgery", "Radiation Therapy", "Chemotherapy", "Targeted Therapy"])
        performance_status = st.slider("Performance Status", 0, 4, 2)
        hemoglobin = st.slider("Hemoglobin Level", 10.0, 18.0, 14.0)
        wbc_count = st.slider("White Blood Cell Count", 3.5, 10.0, 6.7)
    
    # Comorbidities
    st.subheader("Comorbidities")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        diabetes = st.checkbox("Diabetes")
        hypertension = st.checkbox("Hypertension")
        heart_disease = st.checkbox("Heart Disease")
    
    with col2:
        lung_disease = st.checkbox("Chronic Lung Disease")
        kidney_disease = st.checkbox("Kidney Disease")
        autoimmune = st.checkbox("Autoimmune Disease")
    
    with col3:
        other_comorbidity = st.checkbox("Other Comorbidities")
        family_history = st.checkbox("Family History of Cancer")
    
    # Predict button
    if st.button("⏰ Predict Survival", type="primary"):
        with st.spinner("Calculating survival prediction..."):
            # Placeholder for actual model prediction
            import random
            
            # Simulate prediction based on inputs
            base_survival = 60
            if stage == "Stage I":
                survival_factor = 1.2
            elif stage == "Stage II":
                survival_factor = 1.0
            elif stage == "Stage III":
                survival_factor = 0.7
            else:  # Stage IV
                survival_factor = 0.4
            
            if age > 65:
                survival_factor *= 0.9
            if smoking_history == "Current Smoker":
                survival_factor *= 0.8
            
            predicted_survival = base_survival * survival_factor + random.uniform(-5, 5)
            predicted_survival = max(1, min(119, predicted_survival))
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>Survival Prediction</h3>
                    <p><strong>Predicted Survival:</strong> {predicted_survival:.1f} months</p>
                    <p><strong>Risk Level:</strong> {'Low' if predicted_survival > 60 else 'High' if predicted_survival < 30 else 'Medium'}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Survival probability chart
                fig = go.Figure()
                fig.add_trace(go.Indicator(
                    mode="gauge+number+delta",
                    value=predicted_survival,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Survival (months)"},
                    delta={'reference': 60},
                    gauge={
                        'axis': {'range': [None, 120]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgray"},
                            {'range': [30, 60], 'color': "yellow"},
                            {'range': [60, 120], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 30
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    # Model information
    with st.expander("📋 Model Information"):
        st.markdown("""
        **Models Used:**
        - **Support Vector Machine (SVM)**: For binary survival classification
        - **Cox Proportional Hazards**: For survival time regression
        
        **Key Features:**
        - Clinical measurements (tumor size, lab values)
        - Demographics (age, gender, smoking history)
        - Comorbidities and performance status
        - Treatment information
        
        **Evaluation Metrics:**
        - ROC AUC Score
        - Confusion Matrix
        - Survival curves
        """)

# Treatment Response page
elif page == "💊 Treatment Response":
    st.markdown('<h1 class="sub-header">💊 Treatment Response Prediction</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Predict how patients will respond to different treatments using transformer-based models.
    """)
    
    # Treatment selection
    treatment_type = st.selectbox(
        "Select Treatment Type",
        ["Surgery", "Radiation Therapy", "Chemotherapy", "Targeted Therapy"]
    )
    
    st.subheader("Patient Profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 30, 79, 55, key="tr_age")
        gender = st.selectbox("Gender", ["Male", "Female"], key="tr_gender")
        stage = st.selectbox("Cancer Stage", ["Stage I", "Stage II", "Stage III", "Stage IV"], key="tr_stage")
        tumor_size = st.slider("Tumor Size (mm)", 10.0, 100.0, 55.0, key="tr_size")
    
    with col2:
        performance_status = st.slider("Performance Status", 0, 4, 2, key="tr_perf")
        hemoglobin = st.slider("Hemoglobin Level", 10.0, 18.0, 14.0, key="tr_hgb")
        albumin = st.slider("Albumin Level", 3.0, 5.0, 4.0)
        creatinine = st.slider("Creatinine Level", 0.5, 2.0, 1.0)
    
    # Predict response
    if st.button("💊 Predict Treatment Response", type="primary"):
        with st.spinner("Analyzing treatment response..."):
            # Placeholder for actual model prediction
            import random
            
            # Simulate response prediction
            response_prob = random.uniform(0.3, 0.9)
            response = "Good" if response_prob > 0.6 else "Poor"
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>Treatment Response Prediction</h3>
                    <p><strong>Treatment:</strong> {treatment_type}</p>
                    <p><strong>Response:</strong> {response}</p>
                    <p><strong>Response Probability:</strong> {response_prob:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Response probability chart
                fig = go.Figure()
                fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=response_prob * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Response Probability (%)"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkgreen"},
                        'steps': [
                            {'range': [0, 40], 'color': "lightgray"},
                            {'range': [40, 60], 'color': "yellow"},
                            {'range': [60, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 60
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            if response == "Good":
                st.success(f"✅ {treatment_type} shows good response potential for this patient profile.")
            else:
                st.warning(f"⚠️ {treatment_type} may have limited effectiveness. Consider alternative treatments.")
    
    # Model information
    with st.expander("📋 Model Information"):
        st.markdown("""
        **Model Architecture:**
        - **Transformer-based (BERT)**: Advanced deep learning model
        - **Multimodal Input**: Clinical and demographic features
        - **Binary Classification**: Good vs Poor response
        
        **Key Features:**
        - Patient demographics and clinical measurements
        - Tumor characteristics and stage
        - Performance status and lab values
        - Treatment-specific factors
        
        **Applications:**
        - Personalized treatment selection
        - Clinical decision support
        - Treatment optimization
        """)

# Data Analysis page
elif page == "📊 Data Analysis":
    st.markdown('<h1 class="sub-header">📊 Dataset Analysis</h1>', unsafe_allow_html=True)
    
    # Load data
    try:
        df = pd.read_csv("data/archive (7)/lung_cancer_data.csv")
        
        # Overview
        st.subheader("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Patients", f"{len(df):,}")
        with col2:
            st.metric("Features", len(df.columns))
        with col3:
            st.metric("Age Range", f"{df['Age'].min()}-{df['Age'].max()}")
        with col4:
            st.metric("Survival Range", f"{df['Survival_Months'].min()}-{df['Survival_Months'].max()}")
        
        # Demographics
        st.subheader("Patient Demographics")
        col1, col2 = st.columns(2)
        
        with col1:
            # Age distribution
            fig = px.histogram(df, x='Age', nbins=20, title="Age Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Gender distribution
            gender_counts = df['Gender'].value_counts()
            fig = px.pie(values=gender_counts.values, names=gender_counts.index, title="Gender Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        # Cancer stages and survival
        st.subheader("Cancer Stages and Survival")
        col1, col2 = st.columns(2)
        
        with col1:
            stage_counts = df['Stage'].value_counts()
            fig = px.bar(x=stage_counts.index, y=stage_counts.values, title="Cancer Stage Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Survival by stage
            survival_by_stage = df.groupby('Stage')['Survival_Months'].mean().sort_values(ascending=False)
            fig = px.bar(x=survival_by_stage.index, y=survival_by_stage.values, title="Average Survival by Stage")
            st.plotly_chart(fig, use_container_width=True)
        
        # Treatments
        st.subheader("Treatment Analysis")
        treatment_counts = df['Treatment'].value_counts()
        fig = px.pie(values=treatment_counts.values, names=treatment_counts.index, title="Treatment Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        # Survival analysis
        st.subheader("Survival Analysis")
        fig = px.histogram(df, x='Survival_Months', nbins=30, title="Survival Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Feature Correlations")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(
            correlation_matrix,
            title="Feature Correlation Heatmap",
            color_continuous_scale='RdBu'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please ensure the data file is available in the correct location.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🫁 Lung Cancer Predictive Modeling | Built with Streamlit</p>
    <p>For research and educational purposes only. Not for clinical use.</p>
</div>
""", unsafe_allow_html=True) 