# Machine-Failure-Prediction-using-SMOTE-Random-Subsampling-Techniques 



## Project Overview
In this project, we predict machine failures to mitigate potential losses in OEMs(Original Equipment Manufacturers). Heavy machinery is prone to rare but impactful breakdowns, often caused by factors like overstrain, power failure, or tool wear. Predicting these failures beforehand allows industries to address maintenance needs proactively, thereby reducing downtime and associated costs.


---

### Challenges Addressed:

- **Imbalanced Dataset:** Machine failures are rare events. This causes data imbalance, which affects the performance of Machine Learning models. We address this using **SMOTE oversampling** techniques.
- **Failure Categorization:** Identify the specific failure type for better maintenance planning.

---

### Solution:
Using a **Random Forest Classifier**, we:
1. Predict whether a machine will fail (`Target` column).
2. Categorize the type of failure (`Failure Type` column).
3. Provide a user-friendly interaction using **Streamlit**, allowing users to input a machine's `UID` and receive failure predictions.


---

## Features
1. **Predict Machine Failures:** Binary classification to identify machine failure status.
2. **Categorize Failure Type:** Multi-class classification for understanding specific failure causes.
3. **Interactive Dashboard:** User-friendly UI built using Streamlit for easy interaction.
4. **Generate Solution PDF:** Provides a detailed PDF guide to mitigate issues if the machine status indicates a failure.

---

## Installation and Setup Guide

### 1. Clone the Repository
To download the code, open your terminal and run:
```bash
git clone https://github.com/yourusername/machine-failure-prediction.git
cd machine-failure-prediction
```

---

### 2. Install Required Libraries
To streamline the setup, a `requirements.txt` file is included. Install all dependencies by running:
```bash
pip install -r requirements.txt
```

#### Alternatively, the major required libraries include:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `imblearn`
- `joblib`
- `streamlit`

> **Note:** Ensure that you are using **Python 3.7 or higher**.

---

### 3. Download Dataset
The dataset used in this project, `predictive_maintenance.csv`, Download it from [[this link](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)].


---

### 4. Run the Project Locally
This project comes with a **Streamlit UI** for easy interaction. To launch the application:
1. Navigate to the project directory.
2. Run the following command:
   ```bash
   streamlit run app.py
   ```
3. Open the link displayed in the terminal (e.g., `http://localhost:8501`) in your browser to interact with the application.

---

## Workflow Explanation
1. **Dataset Preparation:**
   - Ensure the `predictive_maintenance.csv` file is placed in the project directory.
   - Preprocessing includes handling missing values, encoding categorical variables, and balancing classes using **SMOTE**.

2. **Model Training:**
   - Two **Random Forest Classifier** models are trained:
     - Model 1: Predicts if a machine will fail (`Target`).
     - Model 2: Predicts the type of failure (`Failure Type`).

3. **Model Saving:**
   - Trained models are saved as `model_failure_predictor.pkl` and `model_failure_type_predictor.pkl` for reuse.

4. **Interaction:**
   - Users input a `UID` (machine ID) via the Streamlit interface.
   - The application:
     - Predicts failure status.
     - If a failure is predicted, it categorizes the failure type.

---

## Directory Structure
```
machine-failure-prediction/
│
├── app.py                      # Streamlit application
├── predictive_maintenance.csv  # Dataset (to be added manually)
├── requirements.txt            # Required Python libraries
├── model_failure_predictor.pkl # Trained model for failure prediction
├── model_failure_type_predictor.pkl # Trained model for failure type classification
├── machine_health_model.pkl    # Pre-trained model
├── README.md                   # Documentation
└── preprocessed_data.csv       # Processed dataset
```

---

## Example Predictions
1. Input: `UID = 169`
   - Output: *Machine Failed. Failure Type: Toolware Failure*
2. Input: `UID = 47`
   - Output: *Machine is Working Well.*

---

## Results and Performance
- **Random Forest Classifier Accuracy:**
  - Full features: **99%**
  - UID-based failure type prediction: **94%**
- Balanced dataset achieved using **SMOTE**, ensuring better performance for rare failure events.

