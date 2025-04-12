# 🏡 Housing Price Prediction using Linear Regression

This project predicts house prices using **Linear Regression** based on key features like location, square footage, number of rooms, and amenities such as garage, pool, and garden.

---

## 📊 Dataset

- File: `boston_house_prices.csv`
- Contains features like:
  - `sqft_living` – square footage of the house
  - `bedrooms`, `bathrooms`
  - `has_garage`, `has_pool`, `has_garden`
  - `location` – converted using one-hot encoding
  - `price` – target variable (house price)

---

## 🧠 Workflow

1. **Data Preprocessing**
   - Handle missing values
   - One-hot encoding for categorical `location`
   - Feature and target separation
2. **Model Building**
   - Train-test split (80/20)
   - Model: `LinearRegression` from `sklearn`
3. **Evaluation**
   - Metrics: MSE, MAE, RMSE, R² Score
   - Visualize: Actual vs Predicted scatter plot
4. **Manual Prediction**
   - Using the equation: `y = mx + c`
   - Custom function to predict based on feature input

---

## 🧪 Technologies Used

- Python 🐍
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

---

## ⚙️ How to Run

1. Clone or download this repository
2. Open `housing_price_prediction.ipynb` in **Google Colab** or **Jupyter Notebook**
3. Upload `boston_house_prices.csv` when prompted
4. Run each cell step-by-step

---

## 🧾 Sample Manual Prediction Code

```python
def predict_house_price(model, feature_values):
    m = model.coef_
    c = model.intercept_
    x = np.array(feature_values)
    return np.dot(m, x) + c

# Example input: [sqft, beds, baths, garage, pool, garden, location_Downtown, location_Suburb]
input_data = [3000, 3, 2, 1, 0, 1, 1, 0]
predicted_price = predict_house_price(model, input_data)
print(f"Predicted Price: ₹{predicted_price:.2f}")
