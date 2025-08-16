import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os
from PIL import Image
import io

class LungNoduleDetectionModel:
    def __init__(self):
        self.model = None
        self.class_names = ['Normal', 'Benign', 'Malignant']
        self.input_shape = (256, 256, 1)
        
    def build_model(self):
        """Build the CNN model for lung nodule detection"""
        model = models.Sequential()
        
        # Convolutional layers
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape))
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
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(3, activation='softmax'))
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def preprocess_image(self, image):
        """Preprocess image for prediction"""
        if isinstance(image, str):
            # Load image from file path
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        elif isinstance(image, bytes):
            # Load image from bytes
            img_array = np.frombuffer(image, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        else:
            # Assume PIL Image
            img = np.array(image.convert('L'))
        
        # Resize to 256x256
        img = cv2.resize(img, (256, 256))
        
        # Normalize
        img = img / 255.0
        
        # Add batch and channel dimensions
        img = img.reshape(1, 256, 256, 1)
        
        return img
    
    def predict(self, image):
        """Predict lung nodule classification"""
        if self.model is None:
            raise ValueError("Model not loaded. Please load or train the model first.")
        
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Make prediction
        prediction = self.model.predict(processed_image)
        
        # Get class and confidence
        class_idx = np.argmax(prediction[0])
        confidence = prediction[0][class_idx]
        class_name = self.class_names[class_idx]
        
        return {
            'class': class_name,
            'confidence': float(confidence),
            'probabilities': {
                'Normal': float(prediction[0][0]),
                'Benign': float(prediction[0][1]),
                'Malignant': float(prediction[0][2])
            }
        }
    
    def load_model(self, model_path):
        """Load a trained model"""
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            return True
        return False
    
    def save_model(self, model_path):
        """Save the trained model"""
        if self.model is not None:
            self.model.save(model_path)
            return True
        return False

def load_images_from_folder(folder_path, label):
    """Load images from a folder with given label"""
    images = []
    labels = []
    
    if not os.path.exists(folder_path):
        return np.array(images), np.array(labels)
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (256, 256))
                img = img / 255.0
                images.append(img)
                labels.append(label)
    
    return np.array(images), np.array(labels)

def prepare_dataset(dataset_path):
    """Prepare the dataset for training"""
    # Load images from different classes
    normal_images, normal_labels = load_images_from_folder(
        os.path.join(dataset_path, 'Normal cases'), 0
    )
    benign_images, benign_labels = load_images_from_folder(
        os.path.join(dataset_path, 'Bengin cases'), 1
    )
    malignant_images, malignant_labels = load_images_from_folder(
        os.path.join(dataset_path, 'Malignant cases'), 2
    )
    
    # Combine all images and labels
    X = np.concatenate((normal_images, benign_images, malignant_images), axis=0)
    y = np.concatenate((normal_labels, benign_labels, malignant_labels), axis=0)
    
    # One-hot encoding
    y = to_categorical(y, num_classes=3)
    
    # Reshape for CNN
    X = X.reshape(-1, 256, 256, 1)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test 