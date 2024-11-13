from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import yaml
import os
import pandas as pd
from pathlib import Path
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from datetime import datetime

class ModelTrainer:
    def __init__(self,config_path='Churn/config.yaml'):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing ModelTrainer")
        
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            self.logger.info(f"Loaded configuration from {config_path}")

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Format all paths with timestamp
        self.model_path = self.config['model_paths']['model'].format(date=self.timestamp)
        self.scaler_path = self.config['model_paths']['scaler'].format(date=self.timestamp)
        self.metrics_path = self.config['model_paths']['metrics'].format(date=self.timestamp)
        self.predictions_path = self.config['model_paths']['predictions'].format(date=self.timestamp)    

    def prepare_data(self,df):
        self.logger.info("Starting data preparation")
        features = self.config['model']['features']
        X = df[features]
        y = df['Churn']
        
        self.logger.debug(f"Selected features: {features}")
        self.logger.info(f"Input data shape: X={X.shape}, y={y.shape}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['model']['test_size'],
            random_state=self.config['model']['random_state']
        )
        
        self.logger.info(f"Train-test split completed. Train size: {X_train.shape}, Test size: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        self.logger.info("Starting model training")
        
        scaler = StandardScaler()
        le =LabelEncoder()
        X_train_scaled = scaler.fit_transform(X_train)
        y_train_encoded = le.fit_transform(y_train)
        self.logger.info("Data scaling completed")

        model = RandomForestClassifier(
            n_estimators=self.config['model']['n_estimators'], 
            random_state=self.config["model"]["random_state"]
        )
        self.logger.debug(f"Initialized RandomForestClassifier with {self.config['model']['n_estimators']} estimators")

        model.fit(X_train_scaled, y_train_encoded)
        self.logger.info("Model training completed")

        # Add evaluation on training data
        y_pred = model.predict(X_train_scaled)
        metrics = {
            'accuracy': accuracy_score(y_train_encoded, y_pred),
            'precision': precision_score(y_train_encoded, y_pred),
            'recall': recall_score(y_train_encoded, y_pred),
            'f1': f1_score(y_train_encoded, y_pred)
        }
        
        self.logger.info(f"Training metrics: {metrics}")

        model_path = Path(self.config['model_paths']['model'])
        scaler_path = Path(self.config['model_paths']['scaler'])

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        self.logger.info(f"Saved model to {model_path} and scaler to {scaler_path}")

        return model, scaler, le, metrics

    def predict(self, model, scaler, X):
        """Make predictions on new data"""
        self.logger.info("Making predictions on new data")
        X_scaled = scaler.transform(X)
        
        predictions = {
            'class_prediction': model.predict(X_scaled),
            'probability_scores': model.predict_proba(X_scaled)[:, 1]  # Probability of churn (class 1)
        }
        
        self.logger.info(f"Generated predictions for {len(X)} samples")
        return predictions


    def evaluate_model(self, y_true, y_pred):
        """
        Evaluate model performance and save metrics with timestamp
        """
        # First calculate metrics as single values
        metrics = {
            'metric': ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'],
            'value': [
                accuracy_score(y_true, y_pred),
                precision_score(y_true, y_pred),
                recall_score(y_true, y_pred),
                f1_score(y_true, y_pred),
                roc_auc_score(y_true, y_pred)
            ]
        }
        
        # Create DataFrame with proper structure
        metrics_df = pd.DataFrame(metrics)
        
        # Save metrics
        os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)
        metrics_df.to_csv(self.metrics_path, index=False)
        
        # Return the original metrics dictionary for reference
        return {k: v for k, v in zip(metrics['metric'], metrics['value'])}

    def save_predictions(self, predictions, y_test, filepath="Churn/results/predictions.csv"):
        """
        Save predictions and actual values to a CSV file
        """
        
        
        results_df = pd.DataFrame({
            'actual': y_test,
            'predicted': predictions
        })
        
        os.makedirs(os.path.dirname(self.predictions_path), exist_ok=True)
        results_df.to_csv(self.predictions_path, index=False)
        return self.predictions_path


    def save_models(self, model, scaler):
        """Save model and scaler artifacts"""
        model_path = Path(self.config['model_paths']['model'])
        scaler_path = Path(self.config['model_paths']['scaler'])

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        self.logger.info(f"Saved model to {model_path} and scaler to {scaler_path}")