import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("C:/Users/DELL/Desktop/PROJECTS/crop_recommendation_project/Crop_recommendation.csv")

# Encode target variable
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

# Split features and target
X = df.drop(columns=['label'])
y = df['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(10,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance Plot")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[0])
axes[0].set_title("Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# Feature Importance
sns.barplot(x=feature_importances, y=feature_importances.index, ax=axes[1])
axes[1].set_title("Feature Importance Plot")
axes[1].set_xlabel("Feature Importance")
axes[1].set_ylabel("Features")

plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('C:/Users/DELL/Desktop/PROJECTS/crop_recommendation_project/Crop_recommendation.csv')
df.head()

print("Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nUnique Labels:\n", df['label'].nunique())

label_counts = df['label'].value_counts()
print(label_counts)

plt.figure(figsize=(14,6))
sns.countplot(data=df, x='label', order=label_counts.index)
plt.xticks(rotation=90)
plt.title("Number of Records per Crop Label")
plt.show()

features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

plt.figure(figsize=(18,12))
for i, col in enumerate(features):
    plt.subplot(3, 3, i+1)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

X = df[features]
y = df['label_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

def recommend_crop(input_data):
    input_df = pd.DataFrame([input_data], columns=features)
    pred = model.predict(input_df)
    pred_label = le.inverse_transform(pred)
    return pred_label[0]

sample = {
    'N': 90,
    'P': 42,
    'K': 43,
    'temperature': 25.3,
    'humidity': 80.5,
    'ph': 6.5,
    'rainfall': 100.2
}
recommended = recommend_crop(sample)
print("Recommended Crop:", recommended)

def recommend_crops_multilabel(input_data, threshold=0.05):
    input_df = pd.DataFrame([input_data], columns=features)
    probs = model.predict_proba(input_df)[0]

    # Get indices of crops where probability > threshold
    predicted_indices = [i for i, prob in enumerate(probs) if prob > threshold]
    predicted_crops = le.inverse_transform(predicted_indices)

    # Pair with probabilities
    results = sorted([(crop, round(probs[i]*100, 2)) for i, crop in zip(predicted_indices, predicted_crops)], key=lambda x: -x[1])
    return results

sample_input = {
    'N': 90,
    'P': 42,
    'K': 43,
    'temperature': 25.3,
    'humidity': 80.5,
    'ph': 6.5,
    'rainfall': 100.2
}

multi_recommendations = recommend_crops_multilabel(sample_input, threshold=0.05)
print("Recommended Crops (with confidence > 5%):")
for crop, confidence in multi_recommendations:
    print(f"{crop} - {confidence}%")

def recommend_top_n_crops(input_data, top_n=3):
    input_df = pd.DataFrame([input_data], columns=features)
    probs = model.predict_proba(input_df)[0]

    top_indices = np.argsort(probs)[::-1][:top_n]
    top_crops = le.inverse_transform(top_indices)

    return [(crop, round(probs[i]*100, 2)) for i, crop in zip(top_indices, top_crops)]

recommend_top_n_crops(sample_input, top_n=5)

import joblib

# Save the trained model
joblib.dump(model, 'crop_recommendation_model.pkl')

# Save the label encoder
joblib.dump(le, 'label_encoder.pkl')

# Load the trained model
loaded_model = joblib.load('crop_recommendation_model.pkl')

# Load the label encoder
loaded_le = joblib.load('label_encoder.pkl')

def predict_with_loaded_model(input_data, threshold=0.05):
    input_df = pd.DataFrame([input_data], columns=features)
    probs = loaded_model.predict_proba(input_df)[0]

    predicted_indices = [i for i, prob in enumerate(probs) if prob > threshold]
    predicted_crops = loaded_le.inverse_transform(predicted_indices)

    results = sorted([(crop, round(probs[i]*100, 2)) for i, crop in zip(predicted_indices, predicted_crops)], key=lambda x: -x[1])
    return results

recommend_top_n_crops(sample_input, top_n=5)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Predict on test set
y_pred = model.predict(X_test)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot it
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
fig, ax = plt.subplots(figsize=(16, 16))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=90)
plt.title("Confusion Matrix of Crop Prediction")
plt.savefig("ConfusionMatrix.svg")
plt.show()

# Get feature importances
importances = model.feature_importances_
feature_names = X.columns

# Sort by importance
indices = np.argsort(importances)[::-1]

# Plot
plt.figure(figsize=(10,6))
sns.barplot(x=importances[indices], y=feature_names[indices])
plt.title('Feature Importance in Crop Prediction')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig("FeatureImportance.svg")
plt.show()

from sklearn.metrics import classification_report

# Generate the classification report
report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)

# Convert to DataFrame
report_df = pd.DataFrame(report).transpose()

# Plot as table
fig, ax = plt.subplots(figsize=(14, 12))
ax.axis('off')

table = ax.table(cellText=report_df.round(2).values,
                 colLabels=report_df.columns,
                 rowLabels=report_df.index,
                 loc='center',
                 cellLoc='center')

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.3, 1.3)
plt.title('Classification Report - Crop Prediction', fontsize=16, pad=20)

# Adjust layout to prevent clipping
plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.1)

# Save as SVG
plt.savefig("classification_report.svg", bbox_inches='tight')
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Load dataset
df = pd.read_csv('C:/Users/DELL/Desktop/PROJECTS/crop_recommendation_project/Crop_recommendation.csv')

# Check for missing values
print(df.isnull().sum())

# Features and target variables
X = df.drop('label', axis=1)
y = df['label']

# Encode the labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# One-hot encode labels
y = to_categorical(y, num_classes=len(encoder.classes_))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Build the model
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(encoder.classes_), activation='softmax'))  # Output layer with softmax for multi-class classification

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=128, validation_data=(X_test, y_test))

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy*100:.2f}%')

# Plot the training and validation accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot the training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Predict on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert predictions to class labels
y_true = np.argmax(y_test, axis=1)  # Convert true labels to class labels

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# Plot the importance of the features using the first hidden layer weights
weights = model.layers[0].get_weights()[0]  # Get the weights from the first layer
feature_importance = np.mean(np.abs(weights), axis=1)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(X.columns, feature_importance)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

# Save the model
model.save('crop_recommendation_model.h5')

# Sample input for prediction (e.g., a row from X_test)
sample_input = X_test.iloc[0].values.reshape(1, -1)

# Predict the crop for this input
prediction = model.predict(sample_input)
predicted_class = np.argmax(prediction, axis=1)

# Get the class label from the encoder
predicted_label = encoder.classes_[predicted_class][0]
print(f"Predicted Crop: {predicted_label}")

# Predicting crops with confidence > 5%
threshold = 0.5
predictions = model.predict(X_test)
predicted_probs = np.max(predictions, axis=1)

# Filter predictions with confidence > 5%
high_confidence_predictions = np.where(predicted_probs > threshold)[0]

# Show crops with confidence > 5%
for index in high_confidence_predictions:
    predicted_classes = np.where(predictions[index] > threshold)[0]
    print(f"Record {index}:")
    for class_index in predicted_classes:
        print(f"  Predicted Crop: {encoder.classes_[class_index]} - Confidence: {predictions[index][class_index]*100:.2f}%")

