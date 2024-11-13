import pandas as pd
import numpy as np
from src.utils.logger import Logger



class DataProcessor:
    def __init__(self, config_path='Churn/config.yaml'):
        self.logger = Logger(__name__)
        self.logger.info("Initializing DataProcessor")
        
        self.numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
        self.categorical_columns = ['InternetService', 'Contract', 'PaymentMethod']
        self.transformed_columns = {}
    
    def clean_data(self, df):
        """Clean the raw data"""
        self.logger.info("Starting data cleaning process")
        df_cleaned = df.copy()
        
        # Log initial data stats
        self.logger.info(f"Initial data shape: {df_cleaned.shape}")
        
        # Handle missing values
        missing = df_cleaned.isnull().sum()
        if missing.any():
            self.logger.warning(f"Found missing values:\n{missing[missing > 0]}")
            
        # Convert TotalCharges to numeric
        try:
            df_cleaned['TotalCharges'] = pd.to_numeric(df_cleaned['TotalCharges'], 
                                                     errors='coerce')
            df_cleaned['TotalCharges'].fillna(df_cleaned['MonthlyCharges'], 
                                            inplace=True)
            self.logger.info("Successfully converted TotalCharges to numeric")
        except Exception as e:
            self.logger.error(f"Error converting TotalCharges: {str(e)}")
            raise
            
        self.logger.info("Data cleaning completed")
        return df_cleaned
    
    def transform_features(self, df):
        self.logger.info("Starting feature transformation")
        df_transformed = df.copy()
        
        # Binary encoding
        self.logger.info("Performing binary encoding for gender")
        df_transformed['gender'] = df_transformed['gender'].map({'Female': 0, 'Male': 1})
        
        # Get dummies and track transformations
        self.logger.info("Creating dummy variables for categorical columns")
        for col in self.categorical_columns:
            dummies = pd.get_dummies(df_transformed[col], prefix=col)
            self.transformed_columns[col] = dummies.columns.tolist()
            self.logger.debug(f"Created dummies for {col}: {dummies.columns.tolist()}")
            
        # Apply get_dummies
        df_transformed = pd.get_dummies(
            df_transformed, 
            columns=self.categorical_columns, 
            drop_first=True
        )
        
        self.logger.info(f"Transformation complete. New shape: {df_transformed.shape}")
        return df_transformed
    
    def get_transformed_column_names(self):
        """Get the names of columns after transformation"""
        return self.transformed_columns
    