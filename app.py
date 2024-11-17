# import streamlit as st
# import joblib

# # Load the pre-trained models (replace with your model file paths)
# model_failure = joblib.load(r"C:\\Users\\Rupam Patil\\Downloads\\Machine-Failure-Prediction-main (1)\\Machine-Failure-Prediction-main\\model_failure_predictor.pkl")  # Binary classifier for failure prediction
# model_failure_type = joblib.load(r"C:\\Users\\Rupam Patil\\Downloads\\Machine-Failure-Prediction-main (1)\\Machine-Failure-Prediction-main\\model_failure_type_predictor.pkl")  # Multi-class classifier for failure type prediction

# # Mapping failure type numbers to their corresponding names
# failure_type_mapping = {
#     0: "Heat Dissipation",
#     1: "No Failure",
#     2: "Overstrain",
#     3: "Power Failure",
#     4: "Random",
#     5: "Toolware Failure"
# }

# # Function to predict failure (0 = Working Well, 1 = Failed)
# def predict_failure(model, product_id):
#     # Your logic for predicting failure (ensure input data is in the correct format)
#     # In your case, the product_id is used to query the dataset and predict failure
#     # Replace this with the actual prediction logic
#     return model.predict([[product_id]])[0]

# # Function to predict the failure type if failure occurred
# def predict_failure_type(model, product_id):
#     # Your logic for predicting failure type
#     return model.predict([[product_id]])[0]

# # Streamlit UI
# st.title("Machine Failure Prediction")

# # User input for Product ID
# product_id = st.number_input("Enter Product ID (UDI):", min_value=0)

# if st.button("Predict Machine Status"):
#     if product_id:
#         # Predict whether the machine failed
#         failure = predict_failure(model_failure, product_id)

#         if failure == 0:
#             st.write("Machine is Working Well")
#         else:
#             # Predict the failure type
#             failure_type = predict_failure_type(model_failure_type, product_id)
            
#             # Map the failure type number to its name
#             failure_type_name = failure_type_mapping.get(failure_type, "Unknown Failure Type")
            
#             st.write(f"Machine Failed. Failure Type: {failure_type_name}")
#     else:
#         st.write("Please enter a valid Product ID.")




import streamlit as st
import joblib
import numpy as np

# Load models
model_failure = joblib.load(r"C:\\Users\\Rupam Patil\\Downloads\\Machine-Failure-Prediction-main (1)\\Machine-Failure-Prediction-main\\model_failure_predictor.pkl")  # Binary Classifier
model_failure_type = joblib.load(r"C:\\Users\\Rupam Patil\\Downloads\\Machine-Failure-Prediction-main (1)\\Machine-Failure-Prediction-main\\model_failure_type_predictor.pkl")  # Multi-Class Classifier

# Failure type mapping
failure_mapping = {
    0: "Heat Dissipation",
    1: "No Failure",
    2: "Overstrain",
    3: "Power Failure",
    4: "Random Failure",
    5: "Toolware Failure"
}

# Prediction functions
def predict_failure(model, product_id):
    input_array = np.array([product_id]).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

def predict_failure_type(model, product_id):
    input_array = np.array([product_id]).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

# Streamlit UI
st.title("Machine Failure Prediction and Solutions")
product_id = st.number_input("Enter the UDI (Product ID):", min_value=0, step=1)

if st.button("Predict Machine Status"):
    # Predict failure
    failure = predict_failure(model_failure, product_id)

    if failure == 0:
        st.success("Machine is Working Well")
    else:
        # Predict failure type
        failure_type = predict_failure_type(model_failure_type, product_id)
        failure_name = failure_mapping[failure_type]
        st.error(f"Machine Failed. Failure Type: {failure_name}")
        
        # Show solution PDF
        with open("failure_solutions.pdf", "rb") as pdf_file:
            st.download_button(
                label="📄 Download Failure Solutions PDF",
                data=pdf_file,
                file_name="failure_solutions.pdf",
                mime="application/pdf"
            )
            st.info("Download the PDF to learn how to overcome this failure.")

