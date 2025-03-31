import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
import os
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

data_dir = r'C:\Users\ZZY\Desktop\dopplerdata'
train_dir = os.path.join(data_dir, 'training')
test_dir = os.path.join(data_dir, 'test')

batch_size = 32
img_size = (224, 224)

# ------------------------------------------------------------------------------
# Step 1: Loading data
# ------------------------------------------------------------------------------

class_names = sorted([folder for folder in os.listdir(train_dir) 
                     if os.path.isdir(os.path.join(train_dir, folder))])
num_classes = len(class_names)
print("Class Names:", class_names)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_size,
    color_mode='grayscale',
    label_mode='int'
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=img_size,
    batch_size=batch_size,
    color_mode='grayscale',
    label_mode='int'
)

# ------------------------------------------------------------------------------
# Step 2: 数据预处理（灰度转RGB + 归一化）
# ------------------------------------------------------------------------------

def preprocess(image, label):
    """Preprocess"""
    image = image / 255.0
    image = tf.repeat(image, repeats=3, axis=-1)
    return image, label

train_ds = train_ds.map(preprocess)
test_ds = test_ds.map(preprocess)

# ------------------------------------------------------------------------------
# Step 3: 构建MobileNetV2模型（关键修改点：修复输出层）
# ------------------------------------------------------------------------------

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False 


model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax') 
])

# ------------------------------------------------------------------------------
# Step 4: Training
# ------------------------------------------------------------------------------


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
]


history = model.fit(
    train_ds,
    epochs=100,
    validation_data=test_ds,
    callbacks=callbacks
)


test_images, test_labels = next(iter(test_ds.unbatch().batch(1000)))
y_pred = model.predict(test_images)
y_pred_classes = np.argmax(y_pred, axis=1)


print(classification_report(test_labels, y_pred_classes, target_names=class_names))

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