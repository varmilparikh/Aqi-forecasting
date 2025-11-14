
import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="AQI Predictor Pro", layout="wide")

st.title("🌫️ AQI Prediction App – PRO Version")

@st.cache_data
def load_data():
    return pd.read_csv("city_day.csv")

df = load_data()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

# Detect target column
target_candidates = ["AQI", "AQI_Value", "aqi", "aqi_value"]
target = next((t for t in target_candidates if t in df.columns), None)

# Clean dataset (VERY IMPORTANT)
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=[target])  # target cannot have NaN
df = df.dropna()  # remove rows with NaN in any feature

# Keep only numeric columns
numeric_df = df.select_dtypes(include=["int64", "float64"])

# Final feature list
features = [c for c in numeric_df.columns if c != target]

# Train data
X = df[features]
y = df[target]


# Train model
@st.cache_resource
def train_model():
    model = XGBRegressor(n_estimators=250, learning_rate=0.08, max_depth=6)
    model.fit(X, y)
    joblib.dump(model, "aqi_model.pkl")
    return model

model = train_model()

st.success("✔ Model trained successfully!")

# AQI Category
def aqi_category(aqi):
    if aqi <= 50: return "Good", "🟢"
    elif aqi <= 100: return "Satisfactory", "🟡"
    elif aqi <= 200: return "Moderate", "🟠"
    elif aqi <= 300: return "Poor", "🔴"
    elif aqi <= 400: return "Very Poor", "🟣"
    else: return "Severe", "⚫"

# Prediction Section
st.header("📌 Make a Prediction")

inputs = {}
cols = st.columns(3)
for i, col in enumerate(features):
    with cols[i % 3]:
        inputs[col] = st.number_input(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))

if st.button("Predict AQI"):
    pred = model.predict(pd.DataFrame([inputs]))[0]
    cat, emoji = aqi_category(pred)
    st.metric("Predicted AQI", f"{pred:.2f}")
    st.write(f"### Category: {emoji} **{cat}**")


# Batch Prediction
st.header("📁 Batch CSV Prediction")
uploaded = st.file_uploader("Upload a CSV with same feature columns", type=["csv"])

if uploaded:
    batch = pd.read_csv(uploaded)
    missing = [c for c in features if c not in batch.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
    else:
        preds = model.predict(batch[features])
        batch["Predicted_AQI"] = preds
        st.success("Batch Prediction Completed!")
        st.dataframe(batch.head(), use_container_width=True)
        batch.to_csv("batch_predictions.csv", index=False)
        st.download_button("Download Predictions CSV", data=batch.to_csv(index=False), file_name="predictions.csv")

# Charts
st.header("📈 Visual Insights")

# Correlation heatmap
fig, ax = plt.subplots(figsize=(8, 5))
corr = df[features + [target]].corr()
im = ax.imshow(corr, interpolation="nearest")
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticklabels(corr.columns)
st.pyplot(fig)

# Feature importance
st.subheader("📌 Feature Importance (XGBoost)")

importance = model.feature_importances_
fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.barh(features, importance)
ax2.set_xlabel("Importance Score")
st.pyplot(fig2)

