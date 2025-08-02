# ❤️ Heart Disease Prediction Using Machine Learning

Welcome to the **Heart Disease Prediction** project! This project uses machine learning algorithms to predict the likelihood of heart disease based on patient data.

---

## 📌 Project Description

Build a model that classifies patients as at risk or not for heart disease using healthcare features and ML classification algorithms. 🩺📊

---

## 📂 Dataset Overview

* **Dataset**: Framingham Heart Study Dataset
* **Features**:

  * Age, Gender, Education
  * Smoking habits (currentSmoker, cigsPerDay)
  * Blood pressure (sysBP, diaBP)
  * Cholesterol, glucose, BMI, heartRate
  * Hypertension, Diabetes, Stroke history
* **Target**: `TenYearCHD` (0 = No disease, 1 = Disease)

---

## 🎯 Objectives

* Perform Exploratory Data Analysis (EDA)
* Clean and preprocess healthcare data
* Train a classification model to predict heart disease
* Evaluate using accuracy, confusion matrix, and classification report

---

## 🧰 Technologies Used

* Python 🐍
* Pandas, NumPy
* Seaborn, Matplotlib
* Scikit-learn (Logistic Regression, Evaluation metrics)

---

## 🧪 Workflow Breakdown

### 1️⃣ Data Exploration & Cleaning

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("heart.csv")
df.dropna(inplace=True)
```

### 2️⃣ EDA - Correlation & Target Distribution

```python
sns.heatmap(df.corr(), annot=True)
sns.countplot(data=df, x='TenYearCHD')
```

### 3️⃣ Feature & Label Split

```python
X = df.drop('TenYearCHD', axis=1)
y = df['TenYearCHD']
```

### 4️⃣ Model Training

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

### 5️⃣ Evaluation

```python
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
```

---

## 📊 Visuals & Metrics

* 🔥 Correlation Heatmap of health indicators
  <img width="1088" height="798" alt="image" src="https://github.com/user-attachments/assets/ef912a35-4777-42e1-a21b-8984803895e2" />

* 📋 Confusion Matrix & Precision/Recall/F1-Score
  <img width="961" height="729" alt="image" src="https://github.com/user-attachments/assets/e3da6aee-2368-4eaa-ab0a-cd3c6c963355" />
  <img width="786" height="322" alt="image" src="https://github.com/user-attachments/assets/96e991b7-d7be-44b6-87f4-8945a44909b4" />


---

## 💡 Key Insights

* Logistic Regression gives interpretable results in binary classification
* Data cleaning and handling NaNs is crucial in medical datasets
* Health indicators like BP, BMI, and smoking correlate with CHD

---

## 🧑‍💻 Author

**Nimerta Wadhwani**
💼 AI/ML Intern @ Developers Hub
🔗 [LinkedIn](https://www.linkedin.com/in/nimerta-wadhwani-816362253)

---

## ✅ License

This project is for academic learning purposes only. It is not intended for real medical diagnosis.
