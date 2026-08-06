import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Page Configuration
st.set_page_config(page_title="Boston House Price Predictor", layout="centered")

st.title("🏡 Boston House Price Prediction")
st.write("Enter neighborhood details below to predict estimated house prices.")

# 1. Load Data & Train Model Directly
@st.cache_resource
def train_model():
    # Load dataset
    df = pd.read_csv('boston.csv')
    
    # Strip any trailing/leading spaces in column headers
    df.columns = df.columns.str.strip()
    
    # Find the target column
    target_col = 'MEDV' if 'MEDV' in df.columns else df.columns[-1]

    # Feature & Target Selection
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, list(X.columns)

# Get trained model and feature names
model, feature_names = train_model()

# 2. User Input Form
st.header("Enter Neighborhood Features:")

col1, col2 = st.columns(2)

with col1:
    crim = st.number_input("CRIM (Crime Rate)", min_value=0.0, value=0.1, step=0.01)
    zn = st.number_input("ZN (Residential Land Zoned)", min_value=0.0, value=12.5, step=1.0)
    indus = st.number_input("INDUS (Non-retail Acres)", min_value=0.0, value=11.0, step=0.5)
    chas = st.selectbox("CHAS (Bounds River?)", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    nox = st.number_input("NOX (Nitric Oxides Conc.)", min_value=0.0, value=0.55, step=0.01)
    rm = st.number_input("RM (Avg Rooms/Dwelling)", min_value=1.0, value=6.2, step=0.1)
    age = st.number_input("AGE (Built Prior to 1940 %)", min_value=0.0, value=68.0, step=1.0)

with col2:
    dis = st.number_input("DIS (Distance to Employment)", min_value=0.0, value=3.8, step=0.1)
    rad = st.number_input("RAD (Highway Accessibility)", min_value=1, value=5, step=1)
    tax = st.number_input("TAX (Property Tax Rate)", min_value=100.0, value=400.0, step=10.0)
    ptratio = st.number_input("PTRATIO (Pupil-Teacher Ratio)", min_value=10.0, value=18.5, step=0.5)
    b = st.number_input("B (1000(Bk - 0.63)^2)", min_value=0.0, value=356.0, step=10.0)
    lstat = st.number_input("LSTAT (% Lower Status)", min_value=0.0, value=12.5, step=0.5)

# 3. Prediction Button
if st.button("Predict Price 🚀", use_container_width=True):
    # Construct input dataframe
    input_df = pd.DataFrame([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]],
                            columns=feature_names)
    
    # Generate Prediction
    prediction_raw = model.predict(input_df)[0]
    final_price = prediction_raw * 1000  # MEDV is scaled in $1000s
    
    st.success(f"Estimated House Price: **${final_price:,.2f}**")