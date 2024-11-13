import pandas as pd
import numpy as np
from src.utils.logger import Logger



class FeatureEngineer:
    def __init__(self, data_processor=None):
        self.logger = Logger(__name__)
        self.logger.info("Initializing FeatureEngineer")
        self.data_processor = data_processor
        
    def create_features(self, df):
        self.logger.info("Starting feature creation")
        df_featured = df.copy()
        
        if self.data_processor:
            transformed_cols = self.data_processor.get_transformed_column_names()
            self.logger.debug(f"Transformed columns: {transformed_cols}")
            
            # Calculate service features
            try:
                internet_service_cols = transformed_cols.get('InternetService', [])
                contract_cols = transformed_cols.get('Contract', [])
                service_cols = internet_service_cols + contract_cols
                
                service_cols = [col for col in service_cols if col in df_featured.columns]
                self.logger.info(f"Using service columns: {service_cols}")
                
                if service_cols:
                    df_featured["TotalServices"] = df_featured[service_cols].sum(axis=1)
                    self.logger.info("Successfully created TotalServices feature")
            except Exception as e:
                self.logger.error(f"Error creating service features: {str(e)}")
                raise
        else:
            self.logger.warning("DataProcessor not provided")
            
        # Create tenure bins
        if 'tenure' in df_featured.columns:
            try:
                df_featured['TenureBin'] = pd.qcut(
                    df_featured['tenure'], 
                    q=4, 
                    labels=['0-25%', '25-50%', '50-75%', '75-100%']
                )
                self.logger.info("Successfully created tenure bins")
            except Exception as e:
                self.logger.error(f"Error creating tenure bins: {str(e)}")
                
        self.logger.info(f"Feature engineering complete. Final shape: {df_featured.shape}")
        return df_featured
    