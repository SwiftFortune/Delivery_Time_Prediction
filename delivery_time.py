import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Amazon Delivery Time Predictor", layout="wide")
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
    data.fillna({
        "Agent_Rating": data['Agent_Rating'].mean(),
        "Weather": data['Weather'].mode()[0]
    }, inplace=True)

    # Feature extraction
    data['Order_Hour'] = data['Order_Time'].str[:2].replace('Na', '19').astype(int)
    data['Pickup_Hour'] = data['Pickup_Time'].str[:2].astype(int)

    return data

data = load_data()

# Load trained XGBoost model
@st.cache_resource
def load_model():
    path = r"C:\sachin\Python\amazon project\xgb_delivery_model.pkl"
    try:
        return joblib.load(path)
    except FileNotFoundError:
        st.error("Model file not found! Please check the path.")
        return None

model = load_model()

# Load scaler
@st.cache_resource
def load_scaler():
    path = r"C:\sachin\Python\amazon project\delivery_scaler.pkl"
    try:
        return joblib.load(path)
    except FileNotFoundError:
        st.error("Scaler file not found! Please check the path.")
        return None

scaler = load_scaler()

# Sidebar navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Choose a section:", ["Introduction", "EDA", "Predict Delivery Time", "Self Introduction"])

# ----------------------- SECTION 1: Introduction -----------------------
if option == "Introduction":
    st.markdown("""
    ## 📦 Optimizing Deliveries with AI  
    Welcome to the Amazon Delivery Time Prediction App!

    ✅ **Uncover delivery trends** — See how various factors impact efficiency  
    ✅ **Analyze agent performance** — Track effectiveness in different conditions  
    ✅ **Harness Machine Learning (XGBoost)** for accurate delivery time predictions  

    🔍 Get ready to streamline logistics and enhance customer satisfaction! ⚡  
    """)

# ----------------------- SECTION 2: EDA -----------------------
elif option == "EDA":
    if data is not None:
        st.subheader("📊 Exploratory Data Analysis")
        st.dataframe(data.head())

        st.subheader("📈 Summary Statistics")
        st.dataframe(data.describe())

        st.subheader("⏱️ Delivery Time Distribution")
        fig1, ax1 = plt.subplots()
        sns.histplot(data['Delivery_Time'], bins=30, kde=True, color="skyblue", ax=ax1)
        ax1.set_title("Delivery Time Distribution")
        st.pyplot(fig1)

        st.subheader("📦 Box Plot of Delivery Time")
        fig2, ax2 = plt.subplots()
        sns.boxplot(y=data['Delivery_Time'], ax=ax2, color="lightcoral")
        ax2.set_title("Delivery Time Outliers")
        st.pyplot(fig2)

        st.subheader("🌦️ Average Delivery Time by Weather Conditions")
        weather_avg = data.groupby('Weather')['Delivery_Time'].mean().sort_values()
        fig3, ax3 = plt.subplots()
        sns.barplot(x=weather_avg.index, y=weather_avg.values, palette="flare", ax=ax3)
        ax3.set_title("Weather vs Delivery Time")
        ax3.set_ylabel("Avg Delivery Time (minutes)")
        st.pyplot(fig3)

        st.subheader("🚦 Average Delivery Time by Traffic Conditions")
        traffic_avg = data.groupby('Traffic')['Delivery_Time'].mean().sort_values()
        fig4, ax4 = plt.subplots()
        sns.barplot(x=traffic_avg.index, y=traffic_avg.values, palette="mako", ax=ax4)
        ax4.set_title("Traffic vs Delivery Time")
        ax4.set_ylabel("Avg Delivery Time (minutes)")
        st.pyplot(fig4)

        st.subheader("🚗 Vehicle Type Impact on Delivery Time")
        vehicle_avg = data.groupby('Vehicle')['Delivery_Time'].mean().sort_values()
        fig5, ax5 = plt.subplots()
        sns.barplot(x=vehicle_avg.index, y=vehicle_avg.values, palette="pastel", ax=ax5)
        ax5.set_title("Vehicle Type vs Delivery Time")
        ax5.set_ylabel("Avg Delivery Time (minutes)")
        st.pyplot(fig5)

        st.subheader("🔗 Correlation Between Numerical Features")
        numeric_cols = data.select_dtypes(include=['int64', 'float64'])
        fig6, ax6 = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_cols.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax6)
        ax6.set_title("Feature Correlation Heatmap")
        st.pyplot(fig6)

# ----------------------- SECTION 3: Prediction -----------------------
elif option == "Predict Delivery Time":
    st.subheader("📌 Enter Delivery Details")

    if model is None or data is None or scaler is None:
        st.warning("Model, data, or scaler is missing.")
    else:
        user_input = {}

        with st.form(key="delivery_form"):
            numeric_features = ['Store_Latitude', 'Store_Longitude', 'Drop_Latitude', 'Drop_Longitude']
            categorical_features = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category']
            
            for col in numeric_features:
                user_input[col] = st.number_input(
                    label=col,
                    value=float(data[col].mean())
                )

            for col in categorical_features:
                options = sorted(data[col].astype(str).unique())
                selected = st.selectbox(f"{col}", options + ['Other'], key=col)
                if selected == 'Other':
                    selected = st.text_input(f"Enter custom value for {col}", key=f"{col}_custom")
                user_input[col] = selected

            # Additional time-based features
            user_input["Order_Hour"] = st.slider("Order Hour (0-23)", min_value=0, max_value=23, value=12)
            user_input["Pickup_Hour"] = st.slider("Pickup Hour (0-23)", min_value=0, max_value=23, value=14)

            submit = st.form_submit_button("Submit")

        if submit:
            # Step 1: Convert to DataFrame
            input_df = pd.DataFrame([user_input])

            # Step 2: One-hot encode and align with training features
            input_encoded = pd.get_dummies(input_df)
            input_encoded = input_encoded.reindex(columns=scaler.feature_names_in_, fill_value=0)

            # Step 3: Scale features
            input_scaled = scaler.transform(input_encoded)

            # Step 4: Predict
            prediction = model.predict(input_scaled)

            # Step 5: Display result
            st.success(f"📦 Estimated Delivery Time: **{prediction[0]:.2f} minutes**")


elif option == "Self Introduction":
    st.title("👨‍💻 About Me")
    
    st.markdown("""
    ### Hi, I'm Sachin Hembram 👋
    
    I'm a data science and machine learning enthusiast with a strong interest in solving real-world problems using intelligent systems. This Amazon Delivery Time Predictor is one of my projects aimed at applying machine learning to logistics and customer experience optimization.

    - 🔍 **Expertise**: Python, Machine Learning, Deep Learning, Streamlit, TensorFlow, XGBoost, Scikit-learn
    - 🧠 **Projects**: Voice Gender Prediction, Chest X-ray Classification (TB Detection), Delivery Time Prediction, and more
    - 🌐 **Tech Stack**: pandas, NumPy, matplotlib, seaborn, MLflow, joblib, Streamlit, AWS
    - 🚀 **Goal**: Build end-to-end ML apps that make a real impact and are user-friendly.

    ---
    📫 **Connect with me**   
    - 📧 Email: `sachincmf@gmail.com`    
    """)
