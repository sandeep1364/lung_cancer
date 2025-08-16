# 🫁 Lung Cancer Predictive Modeling - Streamlit App

A comprehensive web application for lung cancer predictive modeling using Streamlit, featuring three main components: Lung Nodule Detection, Survival Prediction, and Treatment Response Prediction.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Required datasets (see Data section below)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Predictive_Modelling_In_Lung_Cancer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the models** (optional - for full functionality)
   ```bash
   python train_models.py
   ```

4. **Run the Streamlit app**
   ```bash
   streamlit run streamlit_app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:8501`

## 📊 Features

### 🏠 Home Dashboard
- Project overview and statistics
- Quick access to all models
- Dataset information and key metrics

### 🔍 Lung Nodule Detection
- **Upload CT scan images** (JPG, PNG format)
- **CNN-based classification** into three categories:
  - Normal: No suspicious findings
  - Benign: Non-cancerous nodules
  - Malignant: Cancerous nodules
- **Confidence scores** and recommendations
- **Real-time analysis** with visual results

### ⏰ Survival Prediction
- **Interactive patient form** with clinical parameters
- **Dual model approach**:
  - Support Vector Machine (SVM) for binary classification
  - Cox Proportional Hazards for survival time regression
- **Risk assessment** with visual gauges
- **Personalized recommendations** based on patient profile

### 💊 Treatment Response Prediction
- **Treatment-specific models** for:
  - Surgery
  - Radiation Therapy
  - Chemotherapy
  - Targeted Therapy
- **Transformer-based architecture** for advanced pattern recognition
- **Response probability** visualization
- **Treatment optimization** suggestions

### 📊 Data Analysis
- **Interactive visualizations** using Plotly
- **Comprehensive dataset exploration**:
  - Patient demographics
  - Cancer stage distribution
  - Survival analysis
  - Treatment patterns
  - Feature correlations
- **Real-time filtering** and exploration

## 🏗️ Architecture

### Project Structure
```
Predictive_Modelling_In_Lung_Cancer/
├── streamlit_app.py              # Main Streamlit application
├── train_models.py               # Model training script
├── requirements.txt              # Python dependencies
├── models/                       # Model modules
│   ├── __init__.py
│   ├── lung_detection_model.py   # CNN for image classification
│   ├── survival_prediction_model.py  # SVM + Cox models
│   └── treatment_response_model.py   # Transformer model
├── data/                         # Dataset directory
│   ├── archive (7)/
│   │   └── lung_cancer_data.csv  # Clinical data
│   └── archive (8)/
│       └── The IQ-OTHNCCD lung cancer dataset/  # CT images
├── trained_models/               # Saved model files
└── README_STREAMLIT.md          # This file
```

### Model Architecture

#### 1. Lung Nodule Detection (CNN)
- **Input**: 256x256 grayscale CT images
- **Architecture**: Convolutional Neural Network
- **Output**: 3-class classification (Normal/Benign/Malignant)
- **Libraries**: TensorFlow, OpenCV

#### 2. Survival Prediction
- **Input**: 38 clinical features
- **Models**: 
  - SVM for binary classification (High/Low survival)
  - Cox Proportional Hazards for survival time
- **Output**: Survival probability and time prediction
- **Libraries**: scikit-learn, lifelines

#### 3. Treatment Response Prediction
- **Input**: Clinical and demographic features
- **Architecture**: Transformer-based neural network
- **Output**: Binary response classification (Good/Poor)
- **Libraries**: PyTorch, transformers

## 📈 Data Requirements

### Clinical Dataset
- **File**: `data/archive (7)/lung_cancer_data.csv`
- **Size**: 23,658 patients with 38 features
- **Features**: Demographics, clinical measurements, comorbidities, lab values
- **Source**: Kaggle - Lung Cancer Prediction Dataset

### Image Dataset
- **Location**: `data/archive (8)/The IQ-OTHNCCD lung cancer dataset/`
- **Size**: 1,097 CT scan images
- **Classes**: Normal (416), Malignant (561), Benign (120)
- **Source**: IQ-OTH/NCCD Lung Cancer Dataset

## 🎯 Usage Examples

### Lung Nodule Detection
1. Navigate to "🔍 Lung Nodule Detection"
2. Upload a CT scan image
3. Click "Analyze Image"
4. Review classification results and confidence scores
5. Follow clinical recommendations

### Survival Prediction
1. Navigate to "⏰ Survival Prediction"
2. Fill in patient information:
   - Demographics (age, gender, smoking history)
   - Clinical parameters (tumor size, stage, lab values)
   - Comorbidities
3. Click "Predict Survival"
4. Review survival probability and risk assessment
5. Consider treatment recommendations

### Treatment Response
1. Navigate to "💊 Treatment Response"
2. Select treatment type
3. Enter patient profile
4. Click "Predict Treatment Response"
5. Review response probability and recommendations

## 🔧 Customization

### Adding New Models
1. Create a new model class in the `models/` directory
2. Implement required methods (`train`, `predict`, `save`, `load`)
3. Add the model to the Streamlit app
4. Update the training script

### Modifying the Interface
- Edit `streamlit_app.py` for UI changes
- Modify CSS styles in the app for visual customization
- Add new pages or features as needed

### Data Integration
- Update data loading functions for new datasets
- Modify preprocessing steps in model classes
- Adjust feature engineering as needed

## 🚨 Important Notes

### Clinical Disclaimer
⚠️ **This application is for research and educational purposes only.**
- Not intended for clinical use
- Results should not replace professional medical advice
- Always consult healthcare professionals for medical decisions

### Model Limitations
- Models are trained on specific datasets
- Performance may vary with different patient populations
- Regular model updates recommended for clinical applications

### Data Privacy
- Ensure compliance with data protection regulations
- Implement appropriate security measures for patient data
- Consider anonymization for sensitive information

## 🛠️ Troubleshooting

### Common Issues

1. **Model not loading**
   - Ensure models are trained: `python train_models.py`
   - Check file paths in `trained_models/` directory

2. **Dataset not found**
   - Verify data files are in correct locations
   - Check file permissions and paths

3. **Dependencies issues**
   - Update pip: `pip install --upgrade pip`
   - Install requirements: `pip install -r requirements.txt`
   - Check Python version compatibility

4. **Streamlit not starting**
   - Check port availability: `streamlit run streamlit_app.py --server.port 8502`
   - Verify firewall settings

### Performance Optimization
- Use GPU acceleration for model training
- Implement model caching for faster predictions
- Consider model quantization for deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the GNU General Public License v3.0.

## 🙏 Acknowledgments

- **Dataset Sources**: Kaggle datasets
- **Medical Expertise**: Professional annotations and validation
- **Open Source**: TensorFlow, PyTorch, scikit-learn, Streamlit communities

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the documentation
3. Open an issue on GitHub
4. Contact the development team

---

**🫁 Empowering healthcare with AI-driven insights for better patient outcomes.** 