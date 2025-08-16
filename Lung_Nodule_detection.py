#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/keethu12345/Predictive_modelling_In_Lung_Cancer/blob/main/Lung_Nodule_detection.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


get_ipython().system('pip install -q kaggle')


# In[ ]:


from google.colab import files
files.upload()  # Upload kaggle.json file


# In[ ]:


get_ipython().system('mkdir -p ~/.kaggle')
get_ipython().system('cp kaggle.json ~/.kaggle/')
get_ipython().system('chmod 600 ~/.kaggle/kaggle.json')


# In[ ]:


get_ipython().system('kaggle datasets download -d adityamahimkar/iqothnccd-lung-cancer-dataset')


# In[ ]:


get_ipython().system('unzip iqothnccd-lung-cancer-dataset.zip')


# In[ ]:


get_ipython().system('ls')


# Identify path of each folder

# In[ ]:


import os

# List the contents of the main directory
main_dir = '/content/'
for root, dirs, _ in os.walk(main_dir):
    print(f"Root: {root}")
    for dir_name in dirs:
        print(f"Dir: {os.path.join(root, dir_name)}")


# ## DATA PREPROCESSING AND SPLITTING
# What this does:
# 1. Loads images from each class folder (Benign, Malignant, Normal).
# 2. Resizes all images to 256x256 pixels and normalizes them.
# 3. Labels the images as 0: Normal, 1: Benign, 2: Malignant.
# 4. One-hot encodes the labels for classification.
# 5. Splits the data into training (80%) and testing (20%) sets.
# 6. Reshapes the images to be compatible with CNN and U-Net.
# 
# 
# 
# 
# 
# 

# In[ ]:


import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Define paths based on your structure
test_cases_path = '/content/Test cases'
dataset_path = '/content/The IQ-OTHNCCD lung cancer dataset/The IQ-OTHNCCD lung cancer dataset'

# Load images and labels
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (256, 256))  # Resize to 256x256
            img = img / 255.0  # Normalize the images
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

# Load images from benign, malignant, and normal folders
benign_images, benign_labels = load_images_from_folder(os.path.join(dataset_path, 'Benign cases'), 1)
malignant_images, malignant_labels = load_images_from_folder(os.path.join(dataset_path, 'Malignant cases'), 2)
normal_images, normal_labels = load_images_from_folder(os.path.join(dataset_path, 'Normal cases'), 0)

# Combine all images and labels
X = np.concatenate((benign_images, malignant_images, normal_images), axis=0)
y = np.concatenate((benign_labels, malignant_labels, normal_labels), axis=0)

# One-hot encoding the labels (0: Normal, 1: Benign, 2: Malignant)
y = to_categorical(y, num_classes=3)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape images for CNN (add channel dimension)
X_train = X_train.reshape(-1, 256, 256, 1)
X_test = X_test.reshape(-1, 256, 256, 1)

print("Data Preprocessing Done. Training data shape:", X_train.shape, "Testing data shape:", X_test.shape)


# ## Building CNN Model

# In[ ]:


import tensorflow as tf
from tensorflow.keras import layers, models

# Build the CNN model
def build_cnn_model():
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten and Dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))  # 3 classes: Normal, Benign, Malignant

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Instantiate and compile the model
cnn_model = build_cnn_model()

# Model summary
cnn_model.summary()


# In[ ]:


# Train the CNN model
history = cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the model
cnn_model.save('lung_nodule_cnn_model.h5')


# ## Building U-Net model for segmentation

# In[ ]:


def unet_model(input_size=(256, 256, 1)):
    inputs = layers.Input(input_size)

    # Contracting path (Encoder)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottleneck
    conv5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)

    # Expanding path (Decoder)
    up6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5)
    merge6 = layers.concatenate([conv4, up6], axis=3)
    conv6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(merge6)
    conv6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
    merge7 = layers.concatenate([conv3, up7], axis=3)
    conv7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(merge7)
    conv7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
    merge8 = layers.concatenate([conv2, up8], axis=3)
    conv8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(merge8)
    conv8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    merge9 = layers.concatenate([conv1, up9], axis=3)
    conv9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(merge9)
    conv9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    # Output layer
    conv10 = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = models.Model(inputs=inputs, outputs=conv10)

    # Compile the model


    return model

# Instantiate and compile the U-Net model
unet = unet_model()

unet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model summary
unet.summary()


# # Training U-Net Model

# In[ ]:


import os
import cv2
import numpy as np

# Load images and masks for segmentation (directly from Test cases folder)
def load_test_images_and_masks(folder):
    images = []
    masks = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (256, 256)) / 255.0  # Resize and normalize the image
            # Create a binary mask based on intensity threshold
            mask = np.where(img > 0.5, 1, 0)  # Assuming the nodules are marked with higher intensities
            images.append(img)
            masks.append(mask)
        else:
            print(f"Warning: Unable to load image {img_path}, skipping.")
    return np.array(images), np.array(masks)

# Path to Test cases folder
test_cases_path = '/content/Test cases'

# Load test images and masks
X_test_images, y_test_masks = load_test_images_and_masks(test_cases_path)

# Check data shapes
print(f"Shape of X_test_images: {X_test_images.shape}")
print(f"Shape of y_test_masks: {y_test_masks.shape}")

# Reshape for U-Net (ensure proper reshaping)
X_test_images = X_test_images.reshape(-1, 256, 256, 1)
y_test_masks = y_test_masks.reshape(-1, 256, 256, 1)

# Check reshaped data
print(f"Reshaped X_test_images: {X_test_images.shape}")
print(f"Reshaped y_test_masks: {y_test_masks.shape}")

# Train U-Net model
unet.fit(X_test_images, y_test_masks, epochs=10, batch_size=8, validation_split=0.2)

# Save the model
unet.save('lung_nodule_unet_model.h5')


# ## Model Evaluation using Performance Metrics

# In[ ]:


from sklearn.metrics import confusion_matrix

# Function to calculate Dice Coefficient
def dice_coefficient(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f))

# Function to calculate IOU (Intersection over Union)
def iou_score(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    return intersection / union

# Function to calculate Sensitivity and Specificity
def sensitivity_specificity(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true_f, y_pred_f).ravel()

    # Sensitivity: tp / (tp + fn)
    sensitivity = tp / (tp + fn)

    # Specificity: tn / (tn + fp)
    specificity = tn / (tn + fp)

    return sensitivity, specificity

# Get model predictions (binary thresholding at 0.5)
y_pred_masks = unet.predict(X_test_images)
y_pred_masks = np.where(y_pred_masks > 0.5, 1, 0)

# Evaluate Dice Coefficient, IOU, Sensitivity, and Specificity
dice = dice_coefficient(y_test_masks, y_pred_masks)
iou = iou_score(y_test_masks, y_pred_masks)
sensitivity, specificity = sensitivity_specificity(y_test_masks, y_pred_masks)

print(f"Dice Coefficient: {dice}")
print(f"IOU Score: {iou}")
print(f"Sensitivity: {sensitivity}")
print(f"Specificity: {specificity}")


# Visualizing ground truth mask vs predicted mask

# In[ ]:


import matplotlib.pyplot as plt

# Function to display a CT scan, the ground truth mask, and the predicted mask
def display_prediction(image, ground_truth_mask, predicted_mask, index):
    plt.figure(figsize=(12, 6))

    # Show the original CT scan image
    plt.subplot(1, 3, 1)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f"CT Scan {index+1}")
    plt.axis('off')

    # Show the ground truth mask
    plt.subplot(1, 3, 2)
    plt.imshow(ground_truth_mask.squeeze(), cmap='gray')
    plt.title(f"Ground Truth Mask {index+1}")
    plt.axis('off')

    # Show the predicted mask
    plt.subplot(1, 3, 3)
    plt.imshow(predicted_mask.squeeze(), cmap='gray')
    plt.title(f"Predicted Mask {index+1}")
    plt.axis('off')

    plt.show()

# Select some samples to visualize
num_samples_to_visualize = 5  # You can increase this number to see more samples
for i in range(num_samples_to_visualize):
    display_prediction(X_test_images[i], y_test_masks[i], y_pred_masks[i], i)


# In[ ]:


# Function to classify multiple images
def classify_images(images, model, num_images=10):
    class_labels = {0: 'Normal', 1: 'Benign', 2: 'Malignant'}

    for i in range(num_images):
        # Select the image
        image = images[i]

        # Reshape the image to (1, 256, 256, 1)
        if len(image.shape) == 2:  # If 2D (H, W), add channel and batch dimensions
            image = image.reshape(1, 256, 256, 1)
        elif len(image.shape) == 3:  # If already (H, W, 1), add batch dimension
            image = np.expand_dims(image, axis=0)

        # Predict the class
        prediction = model.predict(image)
        class_index = np.argmax(prediction, axis=1)[0]
        predicted_label = class_labels[class_index]

        # Display the image and classification result
        plt.figure(figsize=(5, 5))
        plt.imshow(image.squeeze(), cmap='gray')
        plt.title(f"Image {i+1} - Classified as: {predicted_label}")
        plt.axis('off')
        plt.show()

# Classify and display 10 images from the test set
classify_images(X_test_images, cnn_model, num_images=10)

