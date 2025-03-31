import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from tensorflow.keras.applications import EfficientNetB0

# Dataset paths
data_dir = r'C:\Users\ZZY\Desktop\dopplerdata'
train_dir = os.path.join(data_dir, 'training')
test_dir = os.path.join(data_dir, 'test')

# Parameters
img_size = (224, 224)
batch_size = 32

# Load dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir, image_size=img_size, batch_size=batch_size,
    label_mode='int', color_mode='grayscale'
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir, image_size=img_size, batch_size=batch_size,
    label_mode='int', color_mode='grayscale'
)

# Class names
class_names = train_ds.class_names
num_classes = len(class_names)

# Preprocessing
def preprocess(image, label):
    image = tf.image.grayscale_to_rgb(image)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_ds = train_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)

# Load EfficientNetV2 (ViT-like feature extractor)
base_model = EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3),
    pooling='avg'
)
base_model.trainable = True

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

# Build model
model = models.Sequential([
    base_model,
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(train_ds, validation_data=test_ds, epochs=50)

# Evaluate and Confusion Matrix
test_images, test_labels = next(iter(test_ds.unbatch().batch(1000)))
tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
y_pred = model.predict(test_images)
y_pred_classes = np.argmax(y_pred, axis=1)

print(classification_report(test_labels, y_pred_classes, target_names=class_names))

cm = confusion_matrix(test_labels, y_pred_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (ViT-like)')
plt.tight_layout()
plt.show()
