import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Page configuration layout mapping
st.set_page_config(page_title="Iris Classification Dashboard", layout="wide")

st.title("🪻 Iris Species Classification Dashboard")
st.write("Adjust the sliders below to dynamically predict the flower species.")

# Load the dataset properly using Pandas from the local directory
# Make sure 'iris.csv' is saved in the same folder as this script
try:
    df = pd.read_csv('iris.csv')
except FileNotFoundError:
    st.error("Error: 'iris.csv' not found. Please place the CSV file in the same directory.")
    st.stop()

# Separate source features and target vector
X_raw = df.drop(columns=['Species'])
if 'Id' in X_raw.columns:
    X_raw = X_raw.drop(columns=['Id']) # Drop Id column if present
y_target = df['Species']

# Define numerical features explicitly
numerical_features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

# Setup explicit Scikit-Learn Pipeline framework to match feature scaling layout
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features)
    ]
)

@st.cache_resource
def train_production_classifier():
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    model_pipeline.fit(X_raw, y_target)
    return model_pipeline

trained_pipeline = train_production_classifier()

# Sidebar limits automatically sync from real pandas data min/max values
st.sidebar.header("Flower Measurement Features")
input_sepal_len = st.sidebar.slider("Sepal Length (cm)", float(df['SepalLengthCm'].min()), float(df['SepalLengthCm'].max()), float(df['SepalLengthCm'].mean()))
input_sepal_wid = st.sidebar.slider("Sepal Width (cm)", float(df['SepalWidthCm'].min()), float(df['SepalWidthCm'].max()), float(df['SepalWidthCm'].mean()))
input_petal_len = st.sidebar.slider("Petal Length (cm)", float(df['PetalLengthCm'].min()), float(df['PetalLengthCm'].max()), float(df['PetalLengthCm'].mean()))
input_petal_wid = st.sidebar.slider("Petal Width (cm)", float(df['PetalWidthCm'].min()), float(df['PetalWidthCm'].max()), float(df['PetalWidthCm'].mean()))

# Structure user data payload matrix matching the raw features
user_input_raw = pd.DataFrame([{
    'SepalLengthCm': input_sepal_len,
    'SepalWidthCm': input_sepal_wid,
    'PetalLengthCm': input_petal_len,
    'PetalWidthCm': input_petal_wid
}])

# Compute prediction result upon interaction button click
if st.sidebar.button("Predict Iris Species"):
    predicted_specimen = trained_pipeline.predict(user_input_raw)
    st.success(f"Outcome: The model classifies this flower as **{predicted_specimen[0]}**")

st.write("---")
st.subheader("📊 Complete Operational Dataset Workspace Viewer")
st.dataframe(df)
