import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
data = pd.read_csv(r"C:\sachin\Python\amazon Project\amazon_delivery.csv")  # Update path if needed

# Handle missing values
data.fillna({"Agent_Rating": data['Agent_Rating'].mean(), "Weather": data['Weather'].mode()[0]}, inplace=True)

# Feature extraction
data['Order_Hour'] = data['Order_Time'].str[:2].replace('Na', '19').astype(int)
data['Pickup_Hour'] = data['Pickup_Time'].str[:2].astype(int)

# Define features and target
numeric_features = ['Store_Latitude', 'Store_Longitude', 'Drop_Latitude', 'Drop_Longitude']
categorical_features = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category']
target = "Delivery_Time"

# One-hot encode categorical data
data_encoded = pd.get_dummies(data[categorical_features])
final_data = pd.concat([data[numeric_features], data_encoded, data[target]], axis=1)

# Train-test split
X = final_data.drop(columns=[target])
y = final_data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save the trained model and scaler
joblib.dump(model, r"C:\sachin\Python\amazon Project\random_forest_model.pkl")
joblib.dump(scaler, r"C:\sachin\Python\amazon Project\scaler.pkl")

print("Model & Scaler saved successfully!")



import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

st.title("📦 Amazon Delivery Time Prediction")

# Load Data
@st.cache_data
def load_data():
    path = r"C:\sachin\Python\amazon Project\amazon_delivery.csv"
    try:
        data = pd.read_csv(path)
    except FileNotFoundError:
        st.error("Data file not found! Please check the path.")
        return None

    # Handle missing values
    data.fillna({"Agent_Rating": data['Agent_Rating'].mean(), "Weather": data['Weather'].mode()[0]}, inplace=True)

    # Feature extraction
    data['Order_Hour'] = data['Order_Time'].str[:2].replace('Na', '19').astype(int)
    data['Pickup_Hour'] = data['Pickup_Time'].str[:2].astype(int)

    return data

data = load_data()

# Load trained model
@st.cache_resource
def load_model():
    path = r"C:\sachin\Python\amazon Project\random_forest_model.pkl"
    try:
        return joblib.load(path)
    except FileNotFoundError:
        st.error("Model file not found! Please check the path.")
        return None

model = load_model()

st.sidebar.title("Navigation")
option = st.sidebar.radio("Choose a section:", ["Introduction", "EDA", "Predict Delivery Time","Self Introduction"])

# 1️⃣ Introduction Section
if option == "Introduction":
    st.write("""
 📦 **Optimizing Deliveries with AI**  

✅ **Uncover delivery trends** — See how various factors impact efficiency  
✅ **Analyze agent performance** — Track effectiveness in different conditions  
✅ **Harness Machine Learning (Random Forest)** for accurate delivery time predictions  

🔍 Get ready to streamline logistics and enhance customer satisfaction! ⚡  
"""
)

# 2️⃣ Exploratory Data Analysis (EDA)
elif option == "EDA":
    if data is not None:
        st.subheader("📊 Exploratory Data Analysis")
        st.write(data.head())

        st.subheader("📈 Summary Statistics")
        st.dataframe(data.describe())

        # 📌 Delivery Time Distribution
        st.subheader("⏱️ Delivery Time Distribution")
        fig, ax = plt.subplots()
        sns.histplot(data['Delivery_Time'], bins=30, kde=True, color="skyblue", ax=ax)
        ax.set_title("Delivery Time Distribution")
        st.pyplot(fig)

        # 📌 Box Plot – Checking Outliers
        st.subheader("📦 Box Plot of Delivery Time")
        fig, ax = plt.subplots()
        sns.boxplot(y=data['Delivery_Time'], ax=ax, color="lightcoral")
        ax.set_title("Delivery Time Outliers")
        st.pyplot(fig)

        # 📌 Delivery Time vs Weather
        st.subheader("🌦️ Average Delivery Time by Weather Conditions")
        weather_avg = data.groupby('Weather')['Delivery_Time'].mean().sort_values()
        fig, ax = plt.subplots()
        sns.barplot(x=weather_avg.index, y=weather_avg.values, palette="flare", ax=ax)
        ax.set_title("Weather vs Delivery Time")
        ax.set_ylabel("Avg Delivery Time (minutes)")
        ax.set_xlabel("Weather Type")
        st.pyplot(fig)

        # 📌 Delivery Time vs Traffic
        st.subheader("🚦 Average Delivery Time by Traffic Conditions")
        traffic_avg = data.groupby('Traffic')['Delivery_Time'].mean().sort_values()
        fig, ax = plt.subplots()
        sns.barplot(x=traffic_avg.index, y=traffic_avg.values, palette="mako", ax=ax)
        ax.set_title("Traffic vs Delivery Time")
        ax.set_ylabel("Avg Delivery Time (minutes)")
        ax.set_xlabel("Traffic Density")
        st.pyplot(fig)

        # 📌 Delivery Time vs Vehicle Type
        st.subheader("🚗 Vehicle Type Impact on Delivery Time")
        vehicle_avg = data.groupby('Vehicle')['Delivery_Time'].mean().sort_values()
        fig, ax = plt.subplots()
        sns.barplot(x=vehicle_avg.index, y=vehicle_avg.values, palette="pastel", ax=ax)
        ax.set_title("Vehicle Type vs Delivery Time")
        ax.set_ylabel("Avg Delivery Time (minutes)")
        ax.set_xlabel("Vehicle Type")
        st.pyplot(fig)

        # 📌 Correlation Heatmap
        st.subheader("🔗 Correlation Between Numerical Features")
        numeric_cols = data.select_dtypes(include=['int64', 'float64'])
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_cols.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Feature Correlation Heatmap")
        st.pyplot(fig)

        

# 3️⃣ Predict Delivery Time
elif option == "Predict Delivery Time":
    if model is None or data is None:
        st.warning("Model or data is missing.")
    else:
        st.subheader("📌 Enter Delivery Details")
        user_input = {}

        # Load scaler
        scaler = joblib.load(r"C:\sachin\Python\amazon Project\scaler.pkl")

        with st.form(key="delivery_form"):
            numeric_features = ['Store_Latitude', 'Store_Longitude', 'Drop_Latitude', 'Drop_Longitude']
            categorical_features = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category']

            for col in numeric_features:
                user_input[col] = st.number_input(f"{col}", value=float(data[col].mean()))

            for col in categorical_features:
                options = sorted(data[col].astype(str).unique())
                selected = st.selectbox(f"{col}", options + ['Other'], key=col)
                if selected == 'Other':
                    selected = st.text_input(f"Enter custom value for {col}", key=f"{col}_custom")
                user_input[col] = selected

            submit = st.form_submit_button("Submit")

        if submit:
            input_df = pd.DataFrame([user_input])
            input_encoded = pd.get_dummies(input_df)  
            input_encoded = input_encoded.reindex(columns=scaler.feature_names_in_, fill_value=0)
            input_scaled = scaler.transform(input_encoded)

            prediction = model.predict(input_scaled)
            st.success(f"📌 Estimated Delivery Time: {prediction[0]:.2f} minutes")


elif option == "Self Introduction":
    st.subheader("👋 About Me")
    st.write("""
    Hey there! I'm **Sachin**, an aspiring data scientist and a machine learning enthusiast.
    
    - 🚀 Passionate about solving **real-world problems** with AI and analytics  
    - 🤖 Experienced in **Python, Machine Learning, and Deep Learning**  
    - 📊 Love working on projects that **enhance efficiency and optimize workflows**  
    - 💡 Always eager to **learn and share knowledge**   

    Feel free to connect with me and explore this project together! 🎯  
    """)

