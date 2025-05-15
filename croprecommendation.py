# 📦 Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

# 📥 Load the dataset
df = pd.read_csv('data/Crop_recommendation.csv')

# ✅ Feature and label separation
X = df.drop('label', axis=1)   # Features
y = df['label']                # Target (Crop name)

# 🔄 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔃 Feature scaling (optional for tree models, but good for consistency)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🤖 Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 🎯 Evaluate the model
y_pred = model.predict(X_test_scaled)
print("✅ Accuracy Score:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# 💾 Save the model
with open('model/crop_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# 💾 Save the scaler (optional, useful for deployment)
with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# 🔮 Make a sample prediction
# Example: [N, P, K, temperature, humidity, ph, rainfall]
sample = np.array([[90, 40, 40, 25, 80, 6.5, 200]])
scaled_sample = scaler.transform(sample)
prediction = model.predict(scaled_sample)
print("🌾 Recommended Crop:", prediction[0])
