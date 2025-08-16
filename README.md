# Predictive_modelling_In_Lung_Cancer
This repository contains a comprehensive set of machine learning models for lung cancer analysis. The project focuses on three main components: Lung Cancer Detection, Survival Prediction and Treatment Response Prediction.

## Table of Contents
- [Overview](#overview)
- [Components](#components)
  - [Lung Cancer Detection](#lung-cancer-detection)
  - [Survival Prediction](#survival-prediction)
  - [Treatment Response Prediction](#treatment-response-prediction)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

The goal of this project is to develop predictive models that assist in the early detection of lung cancer, evaluate patient survival, and assess treatment responses. By leveraging machine learning techniques, this project aims to contribute to data-driven approaches in oncology, improving patient management and treatment strategies.

## Components

### Lung Cancer Detection
- Implemented using Convolutional Neural Networks (CNNs) for identifying lung cancer in CT scan images.
- The model is trained on IQ-OTH/NCCD - Lung Cancer Dataset and evaluates its performance using various metrics.

### Survival Prediction
- Utilizes the Cox Proportional Hazards model and Support Vector Machines (SVM) to predict patient survival based on clinical features.
- Evaluates the model using metrics such as accuracy, precision, recall, F1 score, and ROC AUC score.

### Treatment Response Prediction
- Employs multimodal transformers to assess treatment responses in lung cancer patients based on various clinical and demographic data.
- Predicts treatment response as a binary outcome and evaluates using appropriate classification metrics.

## Usage
After installing the necessary dependencies, you can run the Colab notebooks located in the main branch. Each notebook contains detailed instructions for executing the code and obtaining results.

### Lung Cancer Detection:
 Open the Lung_Nodule_detection.ipynb notebook to train and evaluate the detection model.
### Survival Prediction:
 Open the Lung_Cancer_Survival_Prediction.ipynb notebook for training and evaluating survival prediction models.
### Treatment Response Prediction:
 Open the Lung_Cancer_Treatment_Response.ipynb notebook for training and evaluating treatment response models.
 
## Evaluation Metrics
The models are evaluated using the following metrics:

### Accuracy: 

Proportion of correctly predicted instances.

### Precision: 

Proportion of true positive predictions to total predicted positives.

### Recall: 

Proportion of true positive predictions to total actual positives.

### F1 Score: 

Harmonic mean of precision and recall.

### ROC AUC Score: 

Measures the model's ability to distinguish between classes.

## Contributing
Contributions are welcome! If you have suggestions for improvements, please create an issue or submit a pull request.

## License
This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.

## Acknowledgments

### Dataset Source: 

 https://www.kaggle.com/datasets/rashadrmammadov/lung-cancer-prediction/data
 https://www.kaggle.com/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset
 
### Libraries Used: 

 TensorFlow, PyTorch, scikit-learn, lifelines, imbalanced-learn, matplotlib, seaborn, sklearn and Transformers-Bert Model.

## Installation

To get started with the project, clone this repository:

```bash
git clone https://github.com/keethu12345/Predictive_Modelling_In_Lung_Cancer.git
cd Predictive_Modelling_In_Lung_Cancer


