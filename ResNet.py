import tensorflow as tf
from tensorflow.keras import layers, Model
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

# ------------------------------------------------------------------------------
# Step 1: Define Dataset Paths
# ------------------------------------------------------------------------------
data_dir = r'C:\Users\ZZY\Desktop\dopplerdata'
train_dir = os.path.join(data_dir, 'training')
test_dir = os.path.join(data_dir, 'test')

# ------------------------------------------------------------------------------
# Step 2: Load Dataset
# ------------------------------------------------------------------------------
batch_size = 32
img_size = (128, 128)  # Smaller size for faster training

# Get class names
class_names = sorted([folder for folder in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, folder))])
num_classes = len(class_names)
print("Class Names:", class_names)

# Load training dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_size,
    color_mode='grayscale',  # Spectrograms are grayscale
    label_mode='int'
)

# Load testing dataset
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=img_size,
    batch_size=batch_size,
    color_mode='grayscale',
    label_mode='int'
)

# ------------------------------------------------------------------------------
# Step 3: Data Preprocessing (Convert Grayscale to RGB + Normalize)
# ------------------------------------------------------------------------------
def preprocess(image, label):
    """Convert grayscale to 3-channel + Normalize"""
    image = image / 255.0  # Normalize to [0,1]
    image = tf.image.grayscale_to_rgb(image)  # Convert grayscale to 3-channel
    return image, label

# Apply preprocessing
train_ds = train_ds.map(preprocess)
test_ds = test_ds.map(preprocess)

# ------------------------------------------------------------------------------
# Step 4: Compute Class Weights for Imbalanced Data
# ------------------------------------------------------------------------------
# Extract labels properly
y_train = np.array([y.numpy() for _, y in train_ds.unbatch()])

if y_train.size > 0:  # Ensure dataset is not empty
    class_weights = compute_class_weight(class_weight='balanced', 
                                         classes=np.unique(y_train), 
                                         y=y_train)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
else:
    print("Warning: No training labels found. Class weights will be ignored.")
    class_weight_dict = None  # Disable class weighting if dataset is empty

print("Class Weights:", class_weight_dict)


# ------------------------------------------------------------------------------
# Step 5: Define CNN + RNN Model
# ------------------------------------------------------------------------------
def build_cnn_rnn_model(num_classes):
    """Build a CNN + RNN model for spectrogram classification."""
    
    inputs = layers.Input(shape=(128, 128, 3))  # Input shape

    # CNN Feature Extractor
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)
    
    # Reshape for RNN Input
    x = layers.Reshape((32, -1))(x)  # Convert to (time_steps, features)

    # RNN Feature Extractor (Bidirectional LSTM)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(32))(x)

    # Fully Connected Layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)  # Final Classification

    return Model(inputs, outputs)

# Instantiate Model
model = build_cnn_rnn_model(num_classes)
model.summary()

# ------------------------------------------------------------------------------
# Step 6: Compile Model
# ------------------------------------------------------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Define Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('best_cnn_rnn_model.h5', save_best_only=True)
]

# ------------------------------------------------------------------------------
# Step 7: Train Model
# ------------------------------------------------------------------------------
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=100,
    callbacks=callbacks,
    class_weight=class_weight_dict  # Apply class weighting
)

# ------------------------------------------------------------------------------
# Step 8: Evaluate Model
# ------------------------------------------------------------------------------
test_images, test_labels = next(iter(test_ds.unbatch().batch(1000)))
y_pred = model.predict(test_images)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification Report
print(classification_report(test_labels, y_pred_classes, target_names=class_names))

# ------------------------------------------------------------------------------
# Step 9: Plot Confusion Matrix
# ------------------------------------------------------------------------------
cm = tf.math.confusion_matrix(test_labels, y_pred_classes)
plt.figure(figsize=(6,5))
sns.heatmap(cm.numpy(), annot=True, fmt='d', 
           xticklabels=class_names, 
           yticklabels=class_names,
           cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
