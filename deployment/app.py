import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

MODEL_REPO_ID = "vsharma15/superkart_mlops_model"
MODEL_FILENAME = "best_superkart_sales_model.joblib"

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Sales Forecasting
st.title("SuperKart Sales Forecasting App")
st.write("Fill the product and store details below to predict total product-store sales.")

product_weight = st.number_input("Product Weight", min_value=0.0, value=12.0, step=0.1)
product_sugar_content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
product_allocated_area = st.number_input("Product Allocated Area", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
product_type = st.selectbox("Product Type", [
    "Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", "Household", "Baking Goods",
    "Snack Foods", "Frozen Foods", "Breakfast", "Health and Hygiene", "Hard Drinks",
    "Canned", "Breads", "Starchy Foods", "Others", "Seafood"
])
product_mrp = st.number_input("Product MRP", min_value=0.0, value=150.0, step=1.0)
store_id = st.text_input("Store Id", value="OUT001")
store_establishment_year = st.number_input("Store Establishment Year", min_value=1900, max_value=2030, value=2000, step=1)
store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
store_location_city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
store_type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Grocery Store"])

input_df = pd.DataFrame({
    "Product_Weight": [product_weight],
    "Product_Sugar_Content": [product_sugar_content],
    "Product_Allocated_Area": [product_allocated_area],
    "Product_Type": [product_type],
    "Product_MRP": [product_mrp],
    "Store_Id": [store_id],
    "Store_Establishment_Year": [store_establishment_year],
    "Store_Size": [store_size],
    "Store_Location_City_Type": [store_location_city_type],
    "Store_Type": [store_type]
})

st.subheader("Input Data")
st.dataframe(input_df)

if st.button("Predict Sales"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Product Store Sales Total: {prediction:,.2f}")
