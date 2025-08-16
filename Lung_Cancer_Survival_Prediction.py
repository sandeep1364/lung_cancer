#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/keethu12345/Predictive_modelling_In_Lung_Cancer/blob/main/Lung_Cancer_Survival_Prediction.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# ### Installations

# In[ ]:


get_ipython().system('pip install lifelines')


# In[ ]:


get_ipython().system('pip install scikit-learn')


# In[ ]:


get_ipython().system('pip install imbalanced-learn')


# ### Loading the datasets

# In[ ]:


from google.colab import files

# Upload the CSV file
uploaded = files.upload()

# After uploading, you can check the uploaded files
import pandas as pd

# Load the dataset into a DataFrame
lung_cancer_data = pd.read_csv('lung_cancer_data.csv')  # Adjust the filename if needed

# Display the first few rows and summary of the dataset
lung_cancer_data_info = lung_cancer_data.info()
lung_cancer_data_head = lung_cancer_data.head()

lung_cancer_data_info, lung_cancer_data_head


# ### Data Preprocessing

# In[ ]:


# Step 2: Data Preprocessing

# Check for missing values
missing_values = lung_cancer_data.isnull().sum()
print("Missing values in each column:")
print(missing_values)

# Remove duplicates
lung_cancer_data = lung_cancer_data.drop_duplicates(subset=['Patient_ID'])

# Create a binary column for survival status
# Assuming that Survival_Months > 0 means the patient died (event)
lung_cancer_data['Survival_Status'] = (lung_cancer_data['Survival_Months'] > 0).astype(int)

# Encode categorical variables
categorical_columns = ['Gender', 'Smoking_History', 'Tumor_Location', 'Stage', 'Treatment', 'Ethnicity', 'Insurance_Type']
lung_cancer_data = pd.get_dummies(lung_cancer_data, columns=categorical_columns, drop_first=True)

# Normalize numerical features
from sklearn.preprocessing import StandardScaler

numerical_columns = ['Age', 'Tumor_Size_mm', 'Performance_Status', 'Blood_Pressure_Systolic',
                     'Blood_Pressure_Diastolic', 'Blood_Pressure_Pulse', 'Hemoglobin_Level',
                     'White_Blood_Cell_Count', 'Platelet_Count', 'Albumin_Level',
                     'Alkaline_Phosphatase_Level', 'Alanine_Aminotransferase_Level',
                     'Aspartate_Aminotransferase_Level', 'Creatinine_Level', 'LDH_Level',
                     'Calcium_Level', 'Phosphorus_Level', 'Glucose_Level',
                     'Potassium_Level', 'Sodium_Level', 'Smoking_Pack_Years']

scaler = StandardScaler()
lung_cancer_data[numerical_columns] = scaler.fit_transform(lung_cancer_data[numerical_columns])

# Display the final dataset structure
print("\nFinal dataset information after preprocessing:")
lung_cancer_data_info = lung_cancer_data.info()


# ### Exploratory Data analysis

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for seaborn
sns.set(style="whitegrid")

# 1. Distribution of Age
plt.figure(figsize=(10, 6))
sns.histplot(lung_cancer_data['Age'], bins=30, kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# 2. Survival Status Count
plt.figure(figsize=(8, 5))
sns.countplot(x='Survival_Status', data=lung_cancer_data)
plt.title('Count of Survival Status')
plt.xlabel('Survival Status (0 = Censored, 1 = Event)')
plt.ylabel('Count')
plt.show()

# 3. Survival by Stage
# Create a new DataFrame to plot survival by stage using the one-hot encoded columns
stage_columns = ['Stage_Stage II', 'Stage_Stage III', 'Stage_Stage IV']
lung_cancer_data['Stage'] = lung_cancer_data[stage_columns].idxmax(axis=1).str.replace('Stage_', '')

plt.figure(figsize=(12, 6))
sns.countplot(x='Stage', hue='Survival_Status', data=lung_cancer_data)
plt.title('Survival Status by Cancer Stage')
plt.xlabel('Cancer Stage')
plt.ylabel('Count')
plt.legend(title='Survival Status', loc='upper right', labels=['Censored', 'Event'])
plt.xticks(rotation=45)
plt.show()

# 4. Correlation Heatmap
# Ensure only numerical columns are included for the heatmap
numerical_cols = lung_cancer_data.select_dtypes(include=['float64', 'int64']).columns
plt.figure(figsize=(12, 8))
correlation_matrix = lung_cancer_data[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# ### Splitting the data

# In[ ]:


from sklearn.model_selection import train_test_split

# Define features and target variable
X = lung_cancer_data.drop(columns=['Patient_ID', 'Survival_Months', 'Survival_Status'])
y = lung_cancer_data['Survival_Status']

# Split the dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Display the shape of the train and test sets
print(f'Training set shape: {X_train.shape}, Test set shape: {X_test.shape}')


# ### Cox Proportional Hazards Model

# In[ ]:


from lifelines import CoxPHFitter
# Check the data types of the columns
print(lung_cancer_data.dtypes)

# Convert relevant columns to numeric where necessary
# Ensure that 'Survival_Months' and 'Survival_Status' are numeric
lung_cancer_data['Survival_Months'] = pd.to_numeric(lung_cancer_data['Survival_Months'], errors='coerce')
lung_cancer_data['Survival_Status'] = pd.to_numeric(lung_cancer_data['Survival_Status'], errors='coerce')

# Remove non-numeric columns from the features
X_train = X_train.select_dtypes(include=[float, int])

# Create a new DataFrame for fitting the Cox model
cox_data_train = X_train.copy()
cox_data_train['Survival_Months'] = lung_cancer_data.loc[X_train.index, 'Survival_Months']
cox_data_train['Survival_Status'] = y_train.values

# Fit the Cox Proportional Hazards model
cph = CoxPHFitter()
cph.fit(cox_data_train, duration_col='Survival_Months', event_col='Survival_Status')

# Display the summary of the model
cph.print_summary()


# ### Evaluating the cox model

# In[ ]:


from lifelines.utils import concordance_index

# 1. Plot the coefficients of the model
plt.figure(figsize=(12, 6))
cph.plot()
plt.title('Cox Proportional Hazards Model Coefficients')
plt.xlabel('Coefficient')
plt.ylabel('Features')
plt.show()

# 2. Evaluate the model using the C-index
# Calculate the predicted hazard ratios
predicted_hazard = cph.predict_partial_hazard(cox_data_train)
# Calculate the C-index
c_index = concordance_index(cox_data_train['Survival_Months'], predicted_hazard, cox_data_train['Survival_Status'])
print(f'C-index: {c_index:.4f}')

# 3. Plot the survival function for an example patient
# Selecting an example patient from the training set
example_patient = cox_data_train.iloc[0]  # Change the index to visualize different patients
survival_function = cph.predict_survival_function(example_patient)

plt.figure(figsize=(10, 6))
plt.step(survival_function.index, survival_function.values, where="post")
plt.title('Predicted Survival Function for Example Patient')
plt.xlabel('Time (Months)')
plt.ylabel('Survival Probability')
plt.grid()
plt.show()


# In[ ]:


survival_probabilities = cph.predict_survival_function(X_test_combined)
print(survival_probabilities)


# ### Support Vector Machine(SVM) Model

# In[ ]:


# Convert all column names to strings
X_combined.columns = X_combined.columns.astype(str)


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load your dataset again if needed (assuming lung_cancer_data is already in the environment)

# Create a binary column for survival status (1 = event, 0 = censored)
lung_cancer_data['Survival_Status'] = (lung_cancer_data['Survival_Months'] > 0).astype(int)

# Prepare the data
X = lung_cancer_data.drop(columns=['Patient_ID', 'Survival_Months', 'Survival_Status'])
y = lung_cancer_data['Survival_Status']

# Check the distribution of the target variable
print(y.value_counts())

# Generate synthetic data for class 0
num_synthetic_samples = 50
synthetic_data = {
    'Age': np.random.uniform(low=X['Age'].min(), high=X['Age'].max(), size=num_synthetic_samples),
    'Tumor_Size_mm': np.random.uniform(low=X['Tumor_Size_mm'].min(), high=X['Tumor_Size_mm'].max(), size=num_synthetic_samples),
    'Performance_Status': np.random.uniform(low=X['Performance_Status'].min(), high=X['Performance_Status'].max(), size=num_synthetic_samples),
    'Blood_Pressure_Systolic': np.random.uniform(low=X['Blood_Pressure_Systolic'].min(), high=X['Blood_Pressure_Systolic'].max(), size=num_synthetic_samples),
    'Blood_Pressure_Diastolic': np.random.uniform(low=X['Blood_Pressure_Diastolic'].min(), high=X['Blood_Pressure_Diastolic'].max(), size=num_synthetic_samples),
    'Blood_Pressure_Pulse': np.random.uniform(low=X['Blood_Pressure_Pulse'].min(), high=X['Blood_Pressure_Pulse'].max(), size=num_synthetic_samples),
    'Hemoglobin_Level': np.random.uniform(low=X['Hemoglobin_Level'].min(), high=X['Hemoglobin_Level'].max(), size=num_synthetic_samples),
    'White_Blood_Cell_Count': np.random.uniform(low=X['White_Blood_Cell_Count'].min(), high=X['White_Blood_Cell_Count'].max(), size=num_synthetic_samples),
    'Platelet_Count': np.random.uniform(low=X['Platelet_Count'].min(), high=X['Platelet_Count'].max(), size=num_synthetic_samples),
    'Albumin_Level': np.random.uniform(low=X['Albumin_Level'].min(), high=X['Albumin_Level'].max(), size=num_synthetic_samples),
    'Alkaline_Phosphatase_Level': np.random.uniform(low=X['Alkaline_Phosphatase_Level'].min(), high=X['Alkaline_Phosphatase_Level'].max(), size=num_synthetic_samples),
    'Alanine_Aminotransferase_Level': np.random.uniform(low=X['Alanine_Aminotransferase_Level'].min(), high=X['Alanine_Aminotransferase_Level'].max(), size=num_synthetic_samples),
    'Aspartate_Aminotransferase_Level': np.random.uniform(low=X['Aspartate_Aminotransferase_Level'].min(), high=X['Aspartate_Aminotransferase_Level'].max(), size=num_synthetic_samples),
    'Creatinine_Level': np.random.uniform(low=X['Creatinine_Level'].min(), high=X['Creatinine_Level'].max(), size=num_synthetic_samples),
    'LDH_Level': np.random.uniform(low=X['LDH_Level'].min(), high=X['LDH_Level'].max(), size=num_synthetic_samples),
    'Calcium_Level': np.random.uniform(low=X['Calcium_Level'].min(), high=X['Calcium_Level'].max(), size=num_synthetic_samples),
    'Phosphorus_Level': np.random.uniform(low=X['Phosphorus_Level'].min(), high=X['Phosphorus_Level'].max(), size=num_synthetic_samples),
    'Glucose_Level': np.random.uniform(low=X['Glucose_Level'].min(), high=X['Glucose_Level'].max(), size=num_synthetic_samples),
    'Potassium_Level': np.random.uniform(low=X['Potassium_Level'].min(), high=X['Potassium_Level'].max(), size=num_synthetic_samples),
    'Sodium_Level': np.random.uniform(low=X['Sodium_Level'].min(), high=X['Sodium_Level'].max(), size=num_synthetic_samples),
    'Smoking_Pack_Years': np.random.uniform(low=X['Smoking_Pack_Years'].min(), high=X['Smoking_Pack_Years'].max(), size=num_synthetic_samples),
}

# Create synthetic DataFrame
synthetic_df = pd.DataFrame(synthetic_data)

# Create target variable for synthetic data (0)
synthetic_df['Survival_Status'] = 0

# Combine the original data with the synthetic data
combined_df = pd.concat([lung_cancer_data, synthetic_df], ignore_index=True)

# Prepare features and target variable
X_combined = combined_df.drop(columns=['Patient_ID', 'Survival_Months', 'Survival_Status'])
y_combined = combined_df['Survival_Status']

# Convert categorical variables to numeric using OneHotEncoder
# Identify categorical columns that may need encoding
categorical_columns = X_combined.select_dtypes(include=['object']).columns.tolist()

# Use OneHotEncoder for categorical features
ohe = OneHotEncoder(sparse_output=False, drop='first')
X_encoded = pd.DataFrame(ohe.fit_transform(X_combined[categorical_columns]))

# Combine with numerical features
X_combined = pd.concat([X_combined.drop(categorical_columns, axis=1).reset_index(drop=True), X_encoded.reset_index(drop=True)], axis=1)

# Convert all column names to strings
X_combined.columns = X_combined.columns.astype(str)

# Split the dataset into training and testing sets (70% train, 30% test)
X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(X_combined, y_combined, test_size=0.3, random_state=42, stratify=y_combined)

# Create and train the SVM model
svm_model = make_pipeline(StandardScaler(), SVC(probability=True))
svm_model.fit(X_train_combined, y_train_combined)

# Make predictions
y_pred_svm = svm_model.predict(X_test_combined)

# Evaluate the SVM model
print("Confusion Matrix:")
print(confusion_matrix(y_test_combined, y_pred_svm))

print("\nClassification Report:")
print(classification_report(y_test_combined, y_pred_svm))

# Calculate ROC AUC score
roc_auc_svm = roc_auc_score(y_test_combined, svm_model.predict_proba(X_test_combined)[:, 1])
print(f'ROC AUC Score: {roc_auc_svm:.4f}')


# ### Evaluating the SVM model and improving the performance

# In[ ]:


from sklearn.model_selection import cross_val_score

# Initialize the model
svm_model_cv = make_pipeline(StandardScaler(), SVC(probability=True))

# Perform k-fold cross-validation (e.g., 5 folds)
cv_scores = cross_val_score(svm_model_cv, X_combined, y_combined, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean CV score:", cv_scores.mean())


# In[ ]:


from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'svc__C': [0.1, 1, 10],
    'svc__kernel': ['linear', 'rbf'],
    'svc__gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train_combined, y_train_combined)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)


# In[ ]:


from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

# Refit the SVM model using the best parameters
best_svm_model = make_pipeline(StandardScaler(), SVC(C=0.1, kernel='linear', gamma='scale', probability=True))
best_svm_model.fit(X_train_combined, y_train_combined)

# Make predictions and evaluate
y_pred_best_svm = best_svm_model.predict(X_test_combined)
print("Confusion Matrix:")
print(confusion_matrix(y_test_combined, y_pred_best_svm))
print("\nClassification Report:")
print(classification_report(y_test_combined, y_pred_best_svm))
roc_auc_best_svm = roc_auc_score(y_test_combined, best_svm_model.predict_proba(X_test_combined)[:, 1])
print(f'ROC AUC Score: {roc_auc_best_svm:.4f}')


# ### Visualization of the model's performances

# A) SVM Model Visualization

# In[ ]:


# Confusion Matrix Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Create confusion matrix
cm = confusion_matrix(y_test_combined, y_pred_svm)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


# In[ ]:


# ROC Curve
from sklearn.metrics import roc_curve

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test_combined, svm_model.predict_proba(X_test_combined)[:, 1])

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = {:.2f})'.format(roc_auc_score(y_test_combined, y_pred_svm)))
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()


# B) COX Model Visualization

# In[ ]:


#Hazard Ratios
# Get the summary dataframe from Cox model
summary = cph.summary

# Plot hazard ratios
plt.figure(figsize=(10, 6))
sns.barplot(x=summary['exp(coef)'], y=summary.index)
plt.axvline(1, linestyle='--', color='red')  # Line at 1
plt.title('Hazard Ratios from Cox Proportional Hazards Model')
plt.xlabel('Hazard Ratio (exp(coef))')
plt.ylabel('Features')
plt.xscale('log')  # Log scale for better visualization
plt.show()


# In[ ]:


#Survival Curves
# Plot survival function for different groups
cph.plot()
plt.title('Survival Function Estimates')
plt.show()

