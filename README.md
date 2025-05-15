 🌾 Crop Recommendation System
 🔍 Overview

This is a machine learning-based crop recommendation system that suggests the most suitable crop to cultivate based on soil and weather parameters. The model is trained using a Random Forest classifier on the publicly available **Crop Recommendation Dataset** from Kaggle.

It takes into account:

* Nitrogen (N)
* Phosphorus (P)
* Potassium (K)
* Temperature (°C)
* Humidity (%)
* pH value of soil
* Rainfall (mm)

🚀 Features

* 📊 **Trained ML Model** with Random Forest
* 🌿 **Label Encoder** to handle crop names
* 🖥️ **Streamlit Web App** for real-time user input and prediction
* 🧠 Predicts the best crop to grow for given conditions
* 📦 Lightweight and easy to deploy

 🛠️ Tech Stack

* Python
* scikit-learn
* Pandas, NumPy
* Streamlit (for UI)
* Pickle (for model serialization)

📁 Project Structure

```
Crop-Recommendation/
├── app.py                  # Streamlit app
├── train_and_save_model.py# Model training + saving script
├── crop_model.pkl          # Trained ML model
├── label_encoder.pkl       # Encoded crop labels
├── Crop_recommendation.csv# Dataset
├── requirements.txt        # Dependencies
└── README.md               # Project info
```

📌 How It Works

1. You input values for soil nutrients and environmental conditions.
2. The app loads the trained model and label encoder.
3. The model predicts the best crop to grow.
4. The app displays the recommendation instantly.

 📉 Dataset

* Source: [Kaggle - Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
* Contains over 2,000 rows with labeled crop types

 🧪 Try It Yourself

To run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```
