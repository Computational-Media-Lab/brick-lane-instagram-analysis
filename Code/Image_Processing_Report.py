#!/usr/bin/env python
# coding: utf-8

# Import Libraries
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from skimage.io import imread
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from scipy.stats import ttest_rel

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Read in data set
df = pd.read_csv('full_data.csv')
# Set path to images folder
img_folder = 'instagram_bricklane_data_output_HTML_files'

##############################################
############### Data Wrangling ###############
##############################################
df = df.drop(df.columns[0], axis = 1)

df = df.drop(['videos', 'videos_name', 'location_name', 'user_website', 'user_bio', 
              'attribution', 'user_full_name', 'user_profile_picture', 'user_username', 'filter',
             'likes', 'comments', 'tags', 'caption_text', 'images', 'link'], axis = 1)

df = df[df['type'] == 'image']

df = df[df['users_in_photo'] >= 0]
df['users_in_photo'] = df['users_in_photo'].astype(int)
df['user_id'] = df['user_id'].astype(int)

df['images_name'] = df['images_name'].str.extract(r'([^?]+\.jpg)')


##############################################
############## Image Processing ##############
##############################################

#---------------------------------
#------------HOG + SVM------------
#---------------------------------

def extract_hog_features(img, resize_shape=(128, 128)):
    img = cv2.resize(img, resize_shape)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hog = cv2.HOGDescriptor(
        _winSize=(resize_shape[0] // 8 * 8, resize_shape[1] // 8 * 8),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )
    features = hog.compute(gray)
    return features.flatten()

X = []
y = []

for idx, row in df.iterrows():
    img_path = os.path.join(img_folder, row['images_name'])
    if not os.path.exists(img_path):
        continue
    img = imread(img_path)
    features = extract_hog_features(img)
    X.append(features)
    y.append(row['users_in_photo']) 

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

svm = LinearSVC()

# Hyperparameter tuning with GridSearchCV
pipe = Pipeline([
    ('scaler', StandardScaler(with_mean = False)),
    ('svm', svm)
])

param_grid = {
    "svm__C": [0.01, 0.1, 1, 10, 100],
    "svm__max_iter": [2000, 5000, 10000]
}

grid = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1, verbose=2)
grid.fit(X_train, y_train)

# Save predictions to df
df['hog_people'] = grid.predict(X)
df['error_hog'] = df['hog_people'] - df['users_in_photo']

filtered_df = df.dropna(subset=['users_in_photo', 'hog_people'])

# Perform t-test to test for significant bias
t_stat, p_val = ttest_rel(filtered_df['hog_people'], filtered_df['users_in_photo'])
print('------------------------------------')
print('---------------OpenCV---------------')
print('------------------------------------')
print(f"T-statistic: {t_stat:.3f}, p-value: {p_val}")

if p_val < 0.05:
    print("Bias detected between predicted and actual counts.")
    print('\n')
else:
    print("No significant bias detected.")
    print('\n')

# Return performance metrics
print(f"Accuracy: {accuracy_score(df['users_in_photo'], df['hog_people'])}")
print(f"Recall: {recall_score(df['users_in_photo'], df['hog_people'], average = 'weighted')}")
print(f"Precision: {precision_score(df['users_in_photo'], df['hog_people'], average = 'weighted')}")
print(f"F1 Score: {f1_score(df['users_in_photo'], df['hog_people'], average = 'weighted')}")
print("\n")

# Print confusion matrix
plt.figure(figsize = (21, 21))
cm = confusion_matrix(df['users_in_photo'], df['hog_people'], labels=range(21))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(21), yticklabels=range(21))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix of People Detection (HOG + SVM)')
plt.savefig("HOG_SVM_heatmap_output.png", dpi = 300, bbox_inches = 'tight')
plt.close()


#----------------------------------
#-----------CNN Model-----------
#----------------------------------

print('------------------------------------')
print('-----------------CNN----------------')
print('------------------------------------')

img_size = (128,128)
X = []
y = []

for idx, row in df.iterrows():
    img_path = os.path.join(img_folder, row['images_name'])
    if not os.path.exists(img_path):
        continue
    img = imread(img_path)
    img = cv2.resize(img, img_size)
    X.append(img)
    y.append(row['users_in_photo'])

X = np.array(X) / 255.0
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(256, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(22, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=32, epochs=50, validation_split = 0.2)

# Predictions
y_pred = np.argmax(model.predict(X_test), axis=1)

# Save predictions back to df
df['cnn_people'] = np.argmax(model.predict(X), axis=1)
df['error_cnn'] = df['cnn_people'] - df['users_in_photo']

filtered_df = df.dropna(subset=['users_in_photo', 'cnn_people'])

# Perform t-test to test for significant bias
t_stat, p_val = ttest_rel(filtered_df['cnn_people'], filtered_df['users_in_photo'])
print(f"T-statistic: {t_stat:.3f}, p-value: {p_val}")

if p_val < 0.05:
    print("Bias detected between predicted and actual counts.")
    print('\n')
else:
    print("No significant bias detected.")
    print('\n')

# Return performance metrics
print(f"Accuracy: {accuracy_score(df['users_in_photo'], df['cnn_people'])}")
print(f"Recall: {recall_score(df['users_in_photo'], df['cnn_people'], average = 'weighted')}")
print(f"Precision: {precision_score(df['users_in_photo'], df['cnn_people'], average = 'weighted')}")
print(f"F1 Score: {f1_score(df['users_in_photo'], df['cnn_people'], average = 'weighted')}")
print("\n")

# Print confusion matrix
plt.figure(figsize = (21, 21))
cm = confusion_matrix(df['users_in_photo'], df['cnn_people'], labels=range(21))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(21), yticklabels=range(21))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix of People Detection (CNN)')
plt.savefig('CNN_heatmap_output.png', dpi = 300, bbox_inches = 'tight')
plt.close()

'''
#----------------------------------
#-----------YOLOv8 Model-----------
#----------------------------------
from ultralytics import YOLO

# Load YOLOv8 model
yolo_model = YOLO('yolov8x.pt') 

# Train with custom hyperparameters
yolo_model.train(data="data.yaml", epochs=200, batch=32, imgsz=1024, lr0=0.01, lrf=0.1, optimizer="SGD")

yolo_counts = []

for idx, row in df.iterrows():
    img_path = os.path.join(img_folder, row['images_name'])
    
    if not os.path.exists(img_path):
        yolo_counts.append(None)
        print("Error: Image path not found")
        continue

    image = imread(img_path)

    try:
        results = yolo_model(img_path, verbose = False, conf = 0.6)[0]
        yolo_people = [
            cls for cls, conf in zip(results.boxes.cls, results.boxes.conf)
            if int(cls) == 0 and conf > 0.6
        ]
        yolo_counts.append(len(yolo_people))
    except:
        yolo_counts.append(None)

df['yolo_people'] = yolo_counts
df['yolo_people'] = df['yolo_people'].astype(int)
df['error_yolo'] = df['yolo_people'] - df['users_in_photo']

filtered_df = df.dropna(subset=['users_in_photo', 'yolo_people'])

# Perform t-test to test for significant bias
t_stat, p_val = ttest_rel(filtered_df['yolo_people'], filtered_df['users_in_photo'])
print('------------------------------------')
print('---------------YOLOv8---------------')
print('------------------------------------')
print(f"T-statistic: {t_stat:.3f}, p-value: {p_val}")

if p_val < 0.05:
    print("Bias detected between predicted and actual counts.")
else:
    print("No significant bias detected.")

# Return performance metrics
print(f"Accuracy: {accuracy_score(df['users_in_photo'], df['yolo_people'])}")
print(f"Recall: {recall_score(df['users_in_photo'], df['yolo_people'], average = 'weighted')}")
print(f"Precision: {precision_score(df['users_in_photo'], df['yolo_people'], average = 'weighted')}")
print(f"F1 Score: {f1_score(df['users_in_photo'], df['yolo_people'], average = 'weighted')}")

# Print confusion matrix
plt.figure(figsize = (21, 21))
cm = confusion_matrix(df['users_in_photo'], df['yolo_people'], labels=range(21))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(21), yticklabels=range(21))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix of People Detection (YOLO)')
plt.show()
'''

#--------------------------------------------
#---------Linear Regression Ensemble---------
#--------------------------------------------

clean_data = [
    (hog, cnn, true)
    for hog, cnn, true in zip(df['hog_people'], df['cnn_people'], df['users_in_photo'])
    if hog is not None and cnn is not None and true is not None
]

hog_vals = [h for h, _, _ in clean_data]
cnn_vals = [y for _, y, _ in clean_data]
true_vals = [t for _, _, t in clean_data]

X = np.column_stack((hog_vals, cnn_vals)) 
y = np.array(true_vals)

reg = LinearRegression().fit(X, y)

print(f"\n--- Strategy: Linear Regression Weighted ---")

y_pred = reg.predict(X)
y_pred_rounded = [round(p) for p in y_pred]

rmse = root_mean_squared_error(y, y_pred_rounded)
acc = accuracy_score(y, y_pred_rounded)
prec = precision_score(y, y_pred_rounded, average = 'weighted')
recall = recall_score(y, y_pred_rounded, average = 'weighted')
f1 = f1_score([1 if t > 0 else 0 for t in y],
    [1 if p > 0 else 0 for p in y_pred_rounded], average = 'weighted')

print(f"Accuracy: {acc}")
print(f"Precision: {prec}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"RMSE: {rmse:.4f}")
print(f"Intercept (β0): {reg.intercept_:.4f}")
print(f"HOG weight (β1): {reg.coef_[0]:.4f}")
print(f"CNN weight (β2): {reg.coef_[1]:.4f}")
print('\n')