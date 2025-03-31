import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Set your dataset path
data_dir = r'C:\Users\ZZY\Desktop\dopplerdata\training'
img_size = (224, 224)

# Read class names
class_names = sorted(os.listdir(data_dir))

# Load images and labels
X, y = [], []
for label, class_name in enumerate(class_names):
    folder = os.path.join(data_dir, class_name)
    for filename in os.listdir(folder):
        if filename.endswith('.png'):
            img_path = os.path.join(folder, filename)
            img = load_img(img_path, target_size=img_size, color_mode='grayscale')
            img_array = img_to_array(img).flatten() / 255.0
            X.append(img_array)
            y.append(label)

X = np.array(X)
y = np.array(y)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Safe PCA component number
max_pca = min(len(X_train), X_train.shape[1], 50)
pca = PCA(n_components=max_pca)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Train SVM
clf = SVC(kernel='rbf', class_weight='balanced')
clf.fit(X_train_pca, y_train)

# Predict
y_pred = clf.predict(X_test_pca)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('SVM Confusion Matrix')
plt.show()

# Classification report
print(classification_report(y_test, y_pred, target_names=class_names))
