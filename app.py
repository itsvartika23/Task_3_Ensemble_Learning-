import streamlit as st
import pandas as pd
import numpy as np
import joblib

#local trained files
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
encoder = joblib.load("encoder.pkl")

st.title("Late Delivery Risk Prediction")
st.write("Enter order details to predict whether the order is at risk of late delivery.")

#numerical features
pca_cols = ['Benefit per order', 'Sales per customer', 'Latitude', 'Longitude',
       'Order Item Discount', 'Order Item Discount Rate',
       'Order Item Product Price', 'Order Item Profit Ratio',
       'Order Item Quantity', 'Sales', 'Order Item Total',
       'Order Profit Per Order', 'Product Price', 'Product Status',
       'Value per Item'
]

#categorical columns
cat_cols = [
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


#user input
st.header("Order Details")

num_values = {}

for col in pca_cols:
    num_values[col] = st.number_input(
        col,
        value=0.0
    )

cat_values = {}

for i, col in enumerate(cat_cols):
    categories = encoder.categories_[i]

    cat_values[col] = st.selectbox(
        col,
        categories
    )


#prediction
if st.button("Predict"):

    # Numerical dataframe
    num_df = pd.DataFrame([num_values])

    #correct cols order
    num_df=num_df[pca_cols]

    # Scale numerical data
    scaled_data = scaler.transform(num_df)

    # PCA transformation
    pca_data = pca.transform(scaled_data)

    # Categorical dataframe
    cat_df = pd.DataFrame([cat_values])

    #correct col order
    cat_df=cat_df[cat_cols]

    # Encode categorical data
    cat_data = encoder.transform(cat_df)

    # Combine PCA + categorical data
    final_input = np.hstack((pca_data, cat_data))

    # Prediction
    prediction = model.predict(final_input)[0]

    # Result
    if prediction == 1:
        st.error(" High Risk of Late Delivery")
    else:
        st.success(" Low Risk of Late Delivery")