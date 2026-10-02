import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA


# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Late Delivery Risk Prediction",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Late Delivery Risk Prediction")
st.write("Predict whether an order is at risk of late delivery.")


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = joblib.load("best_model.pkl")


# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("Data/DataCo.csv")


# ==========================================
# SAME CLEANING AS TRAINING
# ==========================================

df = df.dropna(subset=["Customer Zipcode"])

df = df.drop(columns=["Order Zipcode"])


# ==========================================
# SAME FEATURE ENGINEERING AS TRAINING
# ==========================================

df["Shipping Delay"] = (
    df["Days for shipping (real)"]
    - df["Days for shipment (scheduled)"]
)

df["Value per Item"] = (
    df["Order Item Total"]
    / df["Order Item Quantity"]
)


# ==========================================
# SAME PCA COLUMNS AS TRAINING
# ==========================================

num_cols = df.select_dtypes(include="number").columns

pca_cols = num_cols.drop([
    "Customer Id",
    "Category Id",
    "Product Category Id",
    "Order Item Cardprod Id",
    "Customer Zipcode",
    "Late_delivery_risk",
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Shipping Delay"
])


# ==========================================
# FIT SCALER AND PCA
# SAME AS NOTEBOOK
# ==========================================

X_pca = df[pca_cols]

sc = StandardScaler()

X_scaled = sc.fit_transform(X_pca)


newpca = PCA(n_components=6)

m = newpca.fit_transform(X_scaled)


# ==========================================
# SAME CATEGORICAL COLUMNS AS TRAINING
# ==========================================

cols = [
    "Category Name",
    "Customer Country",
    "Customer City",
    "Customer Segment",
    "Department Name",
    "Market",
    "Order Region",
    "Order Country",
    "Order Status",
    "Shipping Mode"
]


# ==========================================
# FIT ENCODER
# SAME AS NOTEBOOK
# ==========================================

enc = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

cat_data = enc.fit_transform(df[cols])


# ==========================================
# USER INPUT
# ==========================================

st.header("Enter Order Details")


# ------------------------------------------
# CATEGORICAL INPUTS
# ------------------------------------------

st.subheader("Order Information")

cat_input = {}

col1, col2 = st.columns(2)

for i, col in enumerate(cols):

    with col1 if i % 2 == 0 else col2:

        cat_input[col] = st.selectbox(
            col,
            sorted(df[col].dropna().unique())
        )


# ------------------------------------------
# NUMERICAL INPUTS
# ------------------------------------------

st.subheader("Numerical Features")

num_input = {}

hidden_features = ["Latitude", "Longitude"]

visible_pca_cols = [
    col for col in pca_cols
    if col not in hidden_features
]

col1, col2 = st.columns(2)

for i, col in enumerate(visible_pca_cols):

    with col1 if i % 2 == 0 else col2:

        default_value = float(df[col].median())

        num_input[col] = st.number_input(
            col,
            value=default_value
        )

# Keep these values internally for the trained model
num_input["Latitude"] = float(df["Latitude"].median())
num_input["Longitude"] = float(df["Longitude"].median())

# Keep exact training feature order
num_input = {
    col: num_input[col]
    for col in pca_cols
}


# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button("🔮 Predict Late Delivery Risk"):

    # --------------------------------------
    # Create numerical input dataframe
    # --------------------------------------

    numerical_input = pd.DataFrame(
        [num_input]
    )

    # Make sure columns are in same order
    numerical_input = numerical_input[pca_cols]


    # --------------------------------------
    # Apply SAME scaler
    # --------------------------------------

    numerical_scaled = sc.transform(
        numerical_input
    )


    # --------------------------------------
    # Apply SAME PCA
    # --------------------------------------

    numerical_pca = newpca.transform(
        numerical_scaled
    )


    # --------------------------------------
    # Create categorical dataframe
    # --------------------------------------

    categorical_input = pd.DataFrame(
        [cat_input]
    )

    categorical_input = categorical_input[cols]


    # --------------------------------------
    # Apply SAME encoder
    # --------------------------------------

    categorical_encoded = enc.transform(
        categorical_input
    )


    # --------------------------------------
    # Combine PCA + categorical features
    # --------------------------------------

    final_input = np.hstack(
        (
            numerical_pca,
            categorical_encoded
        )
    )


    # --------------------------------------
    # Prediction
    # --------------------------------------

    prediction = model.predict(final_input)[0]


    # --------------------------------------
    # Display result
    # --------------------------------------

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error(
            "⚠️ High Risk: The order is predicted "
            "to have a late delivery."
        )

    else:

        st.success(
            "✅ Low Risk: The order is predicted "
            "not to have a late delivery."
        )