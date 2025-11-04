# --- train_rice_model.py ---
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os
import json

# --- Configuration ---
DATA_DIR = os.path.join(os.getcwd(), 'Data_Small')

# --- Calculate NUM_CLASSES and CLASS_NAMES ---
rice_classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
NUM_CLASSES = len(rice_classes)
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 5
CLASS_NAMES = rice_classes


# --- 1. Data Loading and Augmentation ---
print(f"Detected Classes: {CLASS_NAMES}")
print(f"Number of Classes: {NUM_CLASSES}")

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)


# --- 2. Model Architecture (CNN) ---
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    
    Dense(NUM_CLASSES, activation='softmax') 
])

# --- 3. Compile the Model ---
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

# --- 4. Train the Model ---
print("\nStarting Model Training...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# --- 5. Save the Trained Model and Class Names ---
MODEL_SAVE_PATH = 'rice_cnn_model.h5'
model.save(MODEL_SAVE_PATH)

CLASS_NAMES_PATH = 'rice_class_names.json'
with open(CLASS_NAMES_PATH, 'w') as f:
    json.dump(CLASS_NAMES, f)

print(f"\nTraining Complete. Model saved successfully as: {MODEL_SAVE_PATH}")
print(f"Class names saved as: {CLASS_NAMES_PATH}")