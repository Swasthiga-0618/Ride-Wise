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
 
Feature Importance → shows model interpretability

<img width="606" height="443" alt="Screenshot 2025-10-19 162206" src="https://github.com/user-attachments/assets/65d126fa-699f-4bb7-afda-a7853b8d4ad0" />

<img width="885" height="461" alt="Screenshot 2025-10-19 162713" src="https://github.com/user-attachments/assets/64dc27c3-4a0d-43ac-994c-12cb486dbe49" />


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

Actual vs Predicted → shows accuracy visually

<img width="770" height="480" alt="Screenshot 2025-10-19 162823" src="https://github.com/user-attachments/assets/8c62348a-7c89-40e7-9a24-3e00821e6471" />




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
1. Clone the Repository
```
git clone [your-repo-link]
cd [project-folder]
```
3. Install Dependencies
Make sure you have all required Python libraries installed:
```
pip install -r requirements.txt
```

Note: Typical dependencies include scikit-learn, xgboost, pandas, numpy, matplotlib, seaborn.

3. Load the Trained Model
```
import pickle

#Load the saved stacking model
with open('stacking_model.pkl', 'rb') as f:
    model = pickle.load(f)
```
4. Predict Bike Rentals
```
--> X_test should be your test feature set
y_pred = model.predict(X_test)

print("Predictions:", y_pred[:10])  # Print first 10 predictions
```
5. Evaluate Model Performance
```
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

--> y_test = actual rental counts for test data
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R² Score: {r2:.3f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
```
6. Optional: Visualize Predictions
```
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Rentals")
plt.ylabel("Predicted Rentals")
plt.title("Actual vs Predicted Bike Rentals")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  -->Perfect prediction line
plt.show()
```

# References

XGBoost Documentation: https://xgboost.readthedocs.io/en/stable/

Scikit-learn Stacking Regressor: https://scikit-learn.org/stable/

- - - - - - - - - - - - - - - - - - - - - - - - - - - - -

Author: Swasthiga Sree
swasthiga6@gmail.com
