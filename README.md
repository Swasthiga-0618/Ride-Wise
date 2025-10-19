# Ride-Wise


## 1️⃣ Bike Rental Prediction
This project demonstrates the development of a **Stacking Regressor** model to predict [target variable, e.g., bike rentals, house prices, energy consumption, etc.].  
The goal is to build a robust regression model using **ensemble learning** that can generalize well on unseen data.

---

## 2️⃣ Problem Statement
This project focuses on predicting the daily bike rental count using a stacking ensemble model.

A stacking regressor combines multiple machine learning algorithms — including XGBoost, RandomForest, and Linear Regression — to produce a more accurate and robust prediction. By leveraging the strengths of different models, the ensemble generalizes well to unseen data.

Key Highlights:

Predicts daily bike rentals using historical and environmental features.

Applies preprocessing, feature engineering, and hyperparameter tuning.

Saves the trained model as a pickle file (stacking_model.pkl) for easy reuse.
The dataset contains [number] samples and [number] features representing [context, e.g., daily bike rentals].  
The challenge is to predict the [target] accurately based on multiple input features.

---

## 3️⃣ Dataset

- Shape:  
  - Training: X_train → `(579, 22)`, y_train → `(579,)`  
  - Testing: X_test → `(145, 22)`
  - Target Variable (count → Total number of bikes rented on a particular day)

---

## 4️⃣ Approach / Methodology
1. **Data Preprocessing**  
   - Handling missing values  
   - Encoding categorical features  
   - Scaling numerical features  

2. **Modeling**  
   - Base models: XGBoost, RandomForest, LinearRegression  
   - Stacking Regressor as meta-model  
   - Hyperparameter tuning using `RandomizedSearchCV`  

3. **Evaluation Metrics**  
   - R² Score  
   - Mean Absolute Error (MAE)  
   - Mean Squared Error (MSE)  

---

## 5️⃣ Results
- Test R² Score: 0.862  
- MAE: 475.07  
- MSE: 441190.19  
- Cross-validation R²: 0.8343  

**Sample Predictions:**  
| Actual | Predicted |
|--------|-----------|
| 2500   | 2791.5    |
| 7600   | 7620.6    |
| 6200   | 6176.3    |
| …      | …         |

> The model predicts closely to actual values and generalizes well on unseen data.

---

## 6️⃣ Usage
1. Clone this repository:
```bash
git clone [repo-link]
cd [project-folder]


Install dependencies:

pip install -r requirements.txt


Load the trained model and make predictions:

import pickle

# Load model
with open('stacking_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict
y_pred = model.predict(X_test)


(Optional) Evaluate performance:

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R2 Score: {r2:.3f}, MAE: {mae:.2f}, MSE: {mse:.2f}")



 References

XGBoost Documentation: https://xgboost.readthedocs.io/en/stable/

Scikit-learn Stacking Regressor: https://scikit-learn.org/stable/

- - - - - - - - - - - - - - - - - - - - - - - - - - - - -

Author: Swasthiga Sree
swasthiga6@gmail.com
