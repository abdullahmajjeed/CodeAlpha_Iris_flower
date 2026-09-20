import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Page Configuration (Sab se upar hona chahiye)
st.set_page_config(page_title="Iris Classification Dashboard", layout="wide")
st.title("🪻 Iris Species Classification Dashboard")
st.write("Adjust the sliders below to dynamically predict the flower species.")

# 2. Dataset Ko Load Aur Standardize Karna
try:
    df = pd.read_csv('iris.csv')
    
    # Column ke naamon ko automatic theek karna (Cm hatane ka fix)
    rename_dict = {}
    for col in df.columns:
        if 'SepalLength' in col: rename_dict[col] = 'SepalLengthCm'
        elif 'SepalWidth' in col: rename_dict[col] = 'SepalWidthCm'
        elif 'PetalLength' in col: rename_dict[col] = 'PetalLengthCm'
        elif 'PetalWidth' in col: rename_dict[col] = 'PetalWidthCm'
        elif 'species' in col.lower(): rename_dict[col] = 'Species'
    
    df = df.rename(columns=rename_dict)
    
except FileNotFoundError:
    st.error("❌ Error: 'iris.csv' nahi mili! Meharbani karke check karein ke CSV file isi same folder me ho.")
    st.stop()

# Unnecessary 'Id' column drop karna agar majood ho
X_raw = df.drop(columns=['Species'])
if 'Id' in X_raw.columns:
    X_raw = X_raw.drop(columns=['Id'])

y_target = df['Species']

# 3. Features Name Definition
numerical_features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

# Pipeline Framework Setup
preprocessor = ColumnTransformer(
    transformers=[('num', 'passthrough', numerical_features)]
)

@st.cache_resource
def train_production_classifier():
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    model_pipeline.fit(X_raw, y_target)
    return model_pipeline

# Model train karna
try:
    trained_pipeline = train_production_classifier()
except Exception as e:
    st.error(f"❌ Model training me error aya: {e}")
    st.stop()

# 4. Main UI Screen Sliders Layout
st.subheader("📋 Flower Measurement Features")
col1, col2 = st.columns(2)

with col1:
    input_sepal_len = st.slider("Sepal Length (cm)", float(df['SepalLengthCm'].min()), float(df['SepalLengthCm'].max()), float(df['SepalLengthCm'].mean()))
    input_sepal_wid = st.slider("Sepal Width (cm)", float(df['SepalWidthCm'].min()), float(df['SepalWidthCm'].max()), float(df['SepalWidthCm'].mean()))

with col2:
    input_petal_len = st.slider("Petal Length (cm)", float(df['PetalLengthCm'].min()), float(df['PetalLengthCm'].max()), float(df['PetalLengthCm'].mean()))
    input_petal_wid = st.slider("Petal Width (cm)", float(df['PetalWidthCm'].min()), float(df['PetalWidthCm'].max()), float(df['PetalWidthCm'].mean()))

# Input Data Payload Matrix
user_input_raw = pd.DataFrame([{
    'SepalLengthCm': input_sepal_len,
    'SepalWidthCm': input_sepal_wid,
    'PetalLengthCm': input_petal_len,
    'PetalWidthCm': input_petal_wid
}])

# 5. Live Prediction Result Output
try:
    predicted_specimen = trained_pipeline.predict(user_input_raw)[0]
    prediction_proba = trained_pipeline.predict_proba(user_input_raw)
    max_proba = np.max(prediction_proba) * 100

    st.write("---")
    st.subheader("🎯 Prediction Result")
    st.success(f"Outcome: The model classifies this flower as **{predicted_specimen}** (Confidence: {max_proba:.2f}%)")

except Exception as e:
    st.error(f"❌ Prediction karne me error aya: {e}")

# 6. Complete Data Viewer Table
st.write("---")
st.subheader("📊 Complete Operational Dataset Workspace Viewer")
st.dataframe(df, use_container_width=True)
