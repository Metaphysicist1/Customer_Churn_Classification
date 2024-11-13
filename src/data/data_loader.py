import pandas as pd
import yaml
from pathlib import Path
from src.utils.logger import Logger


class DataLoader:
    def __init__(self, config_path='Churn/config.yaml'):
        self.logger = Logger(__name__)
        self.logger.info("Initializing DataLoader")
        
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            self.logger.info(f"Loaded configuration from {config_path}")

    def load_raw_data(self):
        """Load the raw telco churn dataset"""
        data_path = Path(self.config['data_paths']['raw_data'])
        self.logger.info(f"Loading raw data from {data_path}")
        
        try:
            df = pd.read_csv(data_path)
            self.logger.info(f"Successfully loaded data with shape {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def save_processed_data(self, df):
        output_path = Path(self.config['data_paths']['processed_data'])
        df.to_csv(output_path, index=False)
    