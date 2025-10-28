# 🚚 Amazon Delivery Time Prediction

This project predicts the **delivery time (in minutes)** for Amazon orders based on factors like agent details, location coordinates, weather, traffic, and more. The project includes complete data preprocessing, visualization, model building, evaluation, and tracking with **MLflow**.

---

## 🧠 Project Overview

Timely delivery is a key performance metric in logistics and e-commerce. This project aims to build a **machine learning regression model** that predicts delivery duration based on various input factors to help improve operational efficiency and customer satisfaction.

---

## 🧩 Problem Statement

Given details of Amazon orders, the goal is to **predict the delivery time** (target variable) using features such as:

* Agent’s age and rating
* Store and drop location coordinates
* Weather and traffic conditions
* Vehicle type and area type
* Order and pickup timings

---

## 🗂️ Dataset Details

* **Source:** `amazon_delivery.csv`
* **Rows:** 43,739
* **Columns:** 16

| Feature                    | Description                                    |
| -------------------------- | ---------------------------------------------- |
| Agent_Age                  | Age of the delivery agent                      |
| Agent_Rating               | Rating of the agent                            |
| Store_Latitude / Longitude | Store coordinates                              |
| Drop_Latitude / Longitude  | Delivery destination coordinates               |
| Weather                    | Weather conditions (Sunny, Cloudy, etc.)       |
| Traffic                    | Traffic level (Low, Medium, High, Jam)         |
| Vehicle                    | Type of vehicle used                           |
| Area                       | Urban or Metropolitan                          |
| Delivery_Time              | Target variable – delivery duration in minutes |

---

## 🧹 Data Preprocessing

* Handled missing values using:

  * Mean for numerical columns
  * Mode for categorical columns
* Extracted additional time-based features:

  * `Order_Month`, `Order_Day`, `Order_Hour`, `Pickup_Hour`, etc.
* Removed unnecessary columns like `Order_ID`, `Order_Date`, and `Pickup_Time`.
* Encoded categorical columns using `LabelEncoder`.
* Standardized numerical features with `StandardScaler`.

---

## 📊 Exploratory Data Analysis (EDA)

1. **Delivery Time Distribution** – Boxplots and histograms for outlier detection.
2. **Impact of Weather and Traffic** – Bar charts showing their effect on average delivery time.
3. **Correlation Heatmap** – Visualizing relationships between location coordinates and delivery time.
4. **Agent Performance** – Comparison of delivery time vs. agent rating.

---

## 🤖 Model Building

Multiple regression algorithms were trained and evaluated:

| Model                     | R² Score (%) |
| ------------------------- | ------------ |
| Random Forest Regressor   | **73.50**    |
| XGBoost Regressor         | **74.50**    |
| Decision Tree Regressor   | 50.28        |
| Ridge Regression          | 24.38        |
| Lasso Regression          | 24.15        |
| Elastic Net Regression    | 21.85        |
| Bayesian Ridge Regression | 21.85        |

✅ **Best Model:** `XGBoost Regressor` with **74.5% accuracy**

---

## 📈 Model Evaluation Metrics

* **R² Score:** To measure model performance
* **Mean Squared Error (MSE):** To evaluate prediction errors

Example:

```python
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
```

---

## 🔍 MLflow Tracking

Experiment tracking and model logging done using **MLflow**:

```python
mlflow.set_experiment("Amazon Delivery Time Regression")
mlflow.sklearn.log_model(model, "XGBoost_model")
mlflow.log_metric("R2", r2)
mlflow.log_metric("MSE", mse)
```

Each model’s parameters, metrics, and artifacts are automatically recorded for comparison.

---

## 💾 Model Saving

The best-performing model (`XGBoost`) can be saved for deployment:

```python
import joblib
joblib.dump(xgb_model, "xgboost_delivery_model.pkl")
```

---

## 📉 Visualizations

* **Box Plot:** Distribution of delivery times
* **Bar Chart:** Impact of weather on delivery time
* **Heatmap:** Correlation between drop location and delivery time
* **Box Plot:** Agent performance by rating

---

## 🧰 Technologies Used

* **Python 3.10+**
* **Libraries:**

  * `pandas`, `numpy`, `seaborn`, `matplotlib`
  * `scikit-learn`, `xgboost`, `mlflow`
  * `joblib` for model saving

---

## 🚀 Future Work

* Build a **Streamlit web app** for live delivery time prediction.
* Integrate **real-time weather & traffic APIs** for dynamic predictions.
* Deploy model on **AWS / Hugging Face Spaces / Streamlit Cloud**.

---

## 👨‍💻 Author

**Sachin Hembram**
Data Science | Machine Learning Developer
📧 Email - sachincmf@gmail.com

