import streamlit as st
import pandas as pd
import joblib
import yaml
from pathlib import Path
import logging

class ChurnPredictor:
    def __init__(self, config_path='config.yaml'):
        # Load config
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Load the latest model and scaler
        self.model = self.load_latest_model()
        self.scaler = self.load_latest_scaler()
        self.features = self.config['model']['features']

    def load_latest_model(self):
        model_dir = Path('models/model_artifacts')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        model_path = model_dir / 'random_forest_model.joblib'
        return joblib.load(model_path)

    def load_latest_scaler(self):
        model_dir = Path('models/model_artifacts')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        model_path = model_dir / 'scaler.joblib'
        return joblib.load(model_path)

    def predict(self, input_data):
        # Scale the input data
        scaled_data = self.scaler.transform(input_data)
        # Make prediction
        prediction = self.model.predict(scaled_data)
        probability = self.model.predict_proba(scaled_data)[:, 1]
        return prediction[0], probability[0]

def main():
    st.title('Customer Churn Prediction')
    st.write('Enter customer information to predict churn probability')

    # Initialize predictor
    predictor = ChurnPredictor()

    if st.button('Load High Risk Customer Sample'):
        test_data = {
            'tenure': 2,
            'MonthlyCharges': 90.0,
            'TotalCharges': 180.0,
            'Contract_Month-to-month': 1,
            'PaymentMethod_Electronic check': 1,
            'InternetService_Fiber optic': 1,
            'OnlineSecurity_No': 1,
            'TechSupport_No': 1,
            'StreamingTV_Yes': 1,
            'StreamingMovies_Yes': 1,
            'gender_Male': 1,
            'PaperlessBilling_Yes': 1
        }
        
        # Update session state with test data
        for feature in predictor.features:
            if feature in test_data:
                st.session_state[feature] = test_data[feature]
        
        st.info('High Risk Customer data loaded! Click "Predict Churn" to see the results.')
        st.rerun()  # Force a rerun to update the UI

    # Create input fields using session state
    input_data = {}
    col1, col2 = st.columns(2)
    
    with col1:
        for feature in predictor.features[:len(predictor.features)//2]:
            default_value = st.session_state.get(feature, 0.0)
            input_data[feature] = st.number_input(
                f'Enter {feature}',
                value=default_value,
                key=feature
            )
    
    with col2:
        for feature in predictor.features[len(predictor.features)//2:]:
            default_value = st.session_state.get(feature, 0.0)
            input_data[feature] = st.number_input(
                f'Enter {feature}',
                value=default_value,
                key=feature
            )

    if st.button('Predict Churn'):
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction, probability = predictor.predict(input_df)
        
        # Display results
        st.subheader('Prediction Results')
        if prediction:
            st.error(f'Customer is likely to churn (Probability: {probability:.2%})')
        else:
            st.success(f'Customer is likely to stay (Probability: {1-probability:.2%})')
        
        # Display feature importance
        st.subheader('Feature Importance')
        importance_df = pd.DataFrame({
            'Feature': predictor.features,
            'Importance': predictor.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        st.bar_chart(importance_df.set_index('Feature'))

if __name__ == '__main__':
    main() 