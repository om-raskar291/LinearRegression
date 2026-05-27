# 🏠 House Price Prediction using Linear Regression

## 📌 Overview
This project builds a Linear Regression model to predict house prices using the Boston Housing dataset. It demonstrates the complete machine learning workflow including data analysis, visualization, model building, and evaluation.

---

## 📊 Dataset
The dataset contains multiple features such as:
- Crime rate (crim)
- Number of rooms (rm)
- Property tax (tax)
- Lower status population (lstat)
- Median house price (medv - target)

---

## 🔍 Exploratory Data Analysis (EDA)
- Checked missing values and duplicates
- Analyzed feature distributions using histograms
- Identified relationships using scatter plots
- Used correlation heatmap to find important features
- Detected outliers using boxplots

---

## 🤖 Model Building
- Split data into training and testing sets (80/20)
- Trained Linear Regression model using Scikit-learn
- Generated predictions on test data

---

## 📈 Model Evaluation
- R² Score used to measure performance
- Mean Squared Error (MSE) calculated
- Visualized Actual vs Predicted values

---

## 📊 Key Insights
- Number of rooms (rm) has strong positive impact on price
- Lower status population (lstat) negatively affects price
- Model captures general trend effectively

---

## 🧮 Regression Equation
The model learns a linear equation of the form:

y = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ

---

## 🔮 Custom Prediction
The model can predict house prices for new unseen data.

---

## 🚀 Future Improvements
- Feature scaling
- Outlier treatment
- Regularization (Ridge/Lasso)
- Advanced models (Random Forest, XGBoost)

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn