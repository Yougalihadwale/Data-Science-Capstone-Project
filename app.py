import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the model
loaded_model = pickle.load(open('RandomForest.pkl', 'rb'))

# Load the cleaned data
cleaned_data = pd.read_csv('cleaned data.csv')

# Define categorical columns
category_col = ['brand', 'fuel', 'seller_type', 'transmission', 'owner']

# Function for encoding data
def preprocess_data(df, label_encoders):
    for feature in df.columns:
        if feature in label_encoders:
            df[feature] = label_encoders[feature].transform(df[feature])
    return df

# Load the LabelEncoders used during training
label_encoders = {}
for feature in category_col:
    label_encoder = LabelEncoder()
    label_encoder.fit(cleaned_data[feature])
    label_encoders[feature] = label_encoder

# Title of the app
st.title("Car Selling Price Prediction App")
st.subheader("Please provide the required details to predict the car's selling price.")

st.sidebar.markdown("""
This application predicts the selling price of a car based on various features.
### How to use:
1. **Select the Car Details:** Use the sliders and dropdowns to input the car details.
2. **Predict Price:** Click on the 'Predict Selling Price' button to see the predicted price.
""")

# Display options for data
# display_option = st.radio("Select Display Option:", ["No Data", "Loaded CSV Data", "Encoded Data"])

# Encode the loaded dataset
encoded_data = preprocess_data(cleaned_data.copy(), label_encoders)

# Display the selected data
# if display_option == "No Data":
#     st.subheader("Not displaying either the Loaded CSV File nor the Encoded Data")
# elif display_option == "Loaded CSV Data":
#     st.subheader("Loaded CSV Data:")
#     st.write(cleaned_data)
# elif display_option == "Encoded Data":
#     st.subheader("Encoded Data:")
#     st.write(encoded_data)

# Display sliders for numerical features
km_driven = st.slider("Select KM Driven:", min_value=int(cleaned_data["km_driven"].min()),
                      max_value=int(cleaned_data["km_driven"].max()))
year = st.slider("Select Year:", min_value=int(cleaned_data["year"].min()), max_value=int(cleaned_data["year"].max()))

# Display dropdowns for categorical features
selected_brand = st.selectbox("Select Brand:", cleaned_data["brand"].unique())
brand_filtered_df = cleaned_data[cleaned_data['brand'] == selected_brand]
selected_fuel = st.selectbox("Select Fuel:", cleaned_data["fuel"].unique())
selected_seller_type = st.selectbox("Select Seller Type:", cleaned_data["seller_type"].unique())
selected_transmission = st.selectbox("Select Transmission:", cleaned_data["transmission"].unique())
selected_owner = st.selectbox("Select Owner:", cleaned_data["owner"].unique())

# Create a DataFrame from the user inputs
input_data = pd.DataFrame({
    'brand': [selected_brand],
    'year': [year],
    'km_driven': [km_driven],
    'fuel': [selected_fuel],
    'seller_type': [selected_seller_type],
    'transmission': [selected_transmission],
    'owner': [selected_owner]
})

# st.subheader("Processed Input Data:")
# st.write(input_data)

# Preprocess the user input data using the same label encoders
input_data_encoded = preprocess_data(input_data.copy(), label_encoders)

# st.subheader("Processed Input Data (After Encoding):")
# st.write(input_data_encoded)

# Standardize numerical features using scikit-learn's StandardScaler
scaler = StandardScaler()
numerical_cols = ['year', 'km_driven']
input_data_encoded[numerical_cols] = scaler.fit_transform(input_data_encoded[numerical_cols])

# Make prediction using the loaded model
if st.button("Predict Selling Price"):
    # Make predictions
    predicted_price = loaded_model.predict(input_data_encoded)
    st.subheader("Predicted Selling Price:")
    st.write(f"The predicted selling price is: **_{predicted_price[0]:,.2f}_**")
